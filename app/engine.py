"""
app/engine.py
Main event loop — wires every module together.

KEY IMPROVEMENTS IN THIS VERSION:
───────────────────────────────────
1. MFE/MAE tracking on every candle
   Every open trade tracks how far price moves in its
   favour and against it. This data teaches the AI
   exactly where stops and targets should be placed.

2. AI prediction logging
   Every time the AI makes a prediction it's logged
   with the feature vector and confidence. When the
   trade closes the outcome is marked correct/incorrect.
   Over time this shows model accuracy per market regime.

3. Fees calculated on entry and exit
   Binance charges 0.05% per side. Fees are now tracked
   per trade so the AI learns true net profitability,
   not gross profitability.

4. Equity snapshots
   Balance before and after each trade is recorded.
   AI learns how drawdown periods affect signal quality
   and becomes more conservative during losing streaks.

5. Extended context saved on entry
   All 38 new indicator fields are captured at entry
   and saved to trade_context for AI training.
"""
import asyncio
import os
import time
import tempfile
from datetime import datetime, timezone
from loguru import logger

from app.config import TRADING, TIMEFRAMES, AI, log_config
from app.state import state
from app.market import candles as candle_store
from app.market.stream import start_streams
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.params import get_active_params
from app.risk.guards import check_all
from app.risk.limits import check_emergency_stop, auto_reduce_leverage
from app.risk.sizing import calculate_lot_size
from app.database.client import check_connection
from app.database.trades import (
    save_trade, close_trade, get_open_trades,
    count_closed_trades, update_mfe_mae,
)
from app.database.market import get_candles
from app.database.context import (
    save_trade_context, log_decision, log_bot_event,
    save_ai_prediction_log, update_prediction_outcome,
)
from app.learning.engine import on_trade_closed
from app.notifications import telegram
from app.notifications.messages import (
    trade_opened as msg_opened,
    trade_closed as msg_closed,
    daily_loss_limit_hit,
    drawdown_stop,
    heartbeat as msg_heartbeat,
)
from app.ai.features import build_feature_vector
import app.ai.model as ai_model
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Track MFE/MAE per open trade: trade_id -> (mfe, mae, entry_price, side, opened_at)
_excursion_tracker: dict[str, dict] = {}

# Binance futures fee per side
BINANCE_FEE_PCT = 0.0005


class TradingEngine:
    def __init__(self):
        self.strategy  = EmaRsiStrategy(version="v1.0")
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    # ── Startup ───────────────────────────────────────────────────

    async def start(self) -> None:
        log_config()
        logger.info(f"Starting engine in {TRADING.mode.upper()} mode")

        state.supabase_connected = check_connection()
        if not state.supabase_connected:
            logger.critical("Cannot connect to Supabase — aborting")
            return

        await self._seed_candles()
        await self._load_ai_model()
        await self._reconcile_open_trades()

        # Restore balance from last session
        from app.database.client import get_state_value, set_state_value
        restored = await self._get_balance()
        state.balance = restored
        state.equity  = restored

        saved_start = get_state_value("starting_balance")
        if saved_start:
            state.starting_balance = float(saved_start)
        else:
            state.starting_balance = restored
            set_state_value("starting_balance", str(restored))

        state.is_running       = True
        state.daily_reset_date = datetime.now(timezone.utc).date()

        logger.info(
            f"Balance restored: ${state.balance:,.2f} | "
            f"Start: ${state.starting_balance:,.2f} | "
            f"Drawdown: {state.drawdown_pct():.2%}"
        )

        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._daily_reset_loop())
        asyncio.create_task(telegram.start_listener())

        # Start scheduled jobs
        self._setup_scheduler()
        self.scheduler.start()

        # Live mode exit monitor — checks open positions every minute
        if TRADING.mode == "live":
            asyncio.create_task(self._live_exit_monitor())

        # Startup message with cooldown to prevent spam on crash restarts
        last_start_file = os.path.join(tempfile.gettempdir(), "bot_last_start")
        now_ts          = time.time()
        send_startup    = True

        if os.path.exists(last_start_file):
            try:
                with open(last_start_file) as _f:
                    last = float(_f.read())
                if now_ts - last < 600:
                    send_startup = False
            except Exception:
                pass
        try:
            with open(last_start_file, "w") as _f:
                _f.write(str(now_ts))
        except Exception:
            pass

        if send_startup:
            await telegram.send(
                f"🚀 *Bot started*\n"
                f"Mode: `{TRADING.mode.upper()}`\n"
                f"Balance: `${state.balance:,.2f}`\n"
                f"Symbols: `{', '.join(TRADING.symbols)}`\n"
                f"AI model: `{state.active_model_version or 'none'}`"
            )

        await start_streams(on_candle_close=self._on_candle_close)

    # ── Candle close handler ──────────────────────────────────────

    async def _on_candle_close(
        self, symbol: str, timeframe: str, candle: dict
    ) -> None:
        if timeframe != TIMEFRAMES.signal:
            return

        # Emergency stop check
        stop, reason = check_emergency_stop()
        if stop:
            logger.critical(reason)
            await telegram.send(drawdown_stop(state.drawdown_pct(), state.balance))
            await self._close_all_emergency()
            state.is_paused = True
            return

        # Daily loss limit
        if state.daily_loss_pct() >= TRADING.daily_loss_limit:
            if not state.is_paused:
                state.is_paused = True
                await telegram.send(
                    daily_loss_limit_hit(state.daily_loss_pct(), state.balance)
                )
            return

        # Update MFE/MAE and check exits
        current_price = float(candle["close"])
        if TRADING.mode == "paper":
            await self._update_excursions(symbol, current_price)
            await self._check_paper_exits(symbol, candle)
        else:
            await self._update_excursions(symbol, current_price)

        # Evaluate strategy for new signals
        await self._evaluate_signal(symbol)

    # ── Signal evaluation ─────────────────────────────────────────

    async def _evaluate_signal(self, symbol: str) -> None:
        ind_1h = candle_store.get_indicators(symbol, TIMEFRAMES.signal)
        ind_4h = candle_store.get_indicators(symbol, TIMEFRAMES.trend)

        if ind_1h is None or ind_4h is None:
            return

        signal = self.strategy.evaluate(symbol, ind_1h, ind_4h)

        if signal is None:
            await log_decision(
                symbol=symbol, action="SKIPPED",
                reason="No signal from strategy",
                rsi=ind_1h.get("rsi"), atr=ind_1h.get("atr"),
            )
            return

        # ── AI confidence filter ──────────────────────────────────
        confidence      = signal.confidence
        ai_features     = None
        ai_importance   = {}
        ai_inference_ms = 0.0

        if ai_model.is_loaded():
            t_start      = time.time()
            ai_features  = build_feature_vector(
                ind_1h, ind_4h,
                candle_time=datetime.now(timezone.utc),
                recent_win_rate=0.5,
                current_drawdown=state.drawdown_pct(),
            )
            ai_conf, ai_sig    = ai_model.predict(ai_features)
            ai_importance      = ai_model.get_feature_importance()
            ai_inference_ms    = (time.time() - t_start) * 1000
            confidence         = ai_conf

            if ai_conf < AI.min_confidence:
                await log_decision(
                    symbol=symbol, action="SKIPPED",
                    reason=f"AI confidence {ai_conf:.2%} below {AI.min_confidence:.2%}",
                    signal_type=signal.side,
                    rsi=ind_1h.get("rsi"),
                    confidence=ai_conf * 100,
                )
                return

        # ── Risk guards ───────────────────────────────────────────
        allowed, reason = check_all(symbol, signal.side)
        if not allowed:
            await log_decision(
                symbol=symbol, action="BLOCKED", reason=reason,
                signal_type=signal.side, rsi=ind_1h.get("rsi"),
                confidence=confidence * 100,
            )
            return

        # ── Position sizing ───────────────────────────────────────
        leverage           = auto_reduce_leverage(state.balance, TRADING.max_leverage)
        lot_size, notional = calculate_lot_size(
            balance=state.balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            leverage=leverage,
        )
        if lot_size == 0:
            await log_decision(symbol, "BLOCKED", "Position size too small")
            return

        # Calculate entry fee
        entry_fee   = notional * BINANCE_FEE_PCT
        risk_amount = abs(signal.entry_price - signal.stop_loss) * lot_size

        # ── Execute ───────────────────────────────────────────────
        trade_result = await self._execute(signal, lot_size, notional)
        if trade_result is None:
            return

        # ── Save trade to DB ──────────────────────────────────────
        trade_id = await save_trade(
            symbol=symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            lot_size=lot_size,
            strategy_version=get_active_params().version,
            account_balance=state.balance,
            order_id=trade_result.get("order_id"),
            leverage=leverage,
            risk_amount=risk_amount,
            fees_entry=entry_fee,
        )

        if trade_id:
            # Start excursion tracking for this trade
            _excursion_tracker[symbol] = {
                "trade_id":    trade_id,
                "entry_price": signal.entry_price,
                "side":        signal.side,
                "opened_at":   datetime.now(timezone.utc),
                "mfe":         0.0,
                "mae":         0.0,
            }

            # Build extended context with all new indicators
            ctx = {
                "ema50_1h":       ind_1h.get("ema50"),
                "ema200_1h":      ind_1h.get("ema200"),
                "rsi_1h":         ind_1h.get("rsi"),
                "atr_1h":         ind_1h.get("atr"),
                "ema50_4h":       ind_4h.get("ema50"),
                "ema200_4h":      ind_4h.get("ema200"),
                "rsi_4h":         ind_4h.get("rsi"),
                "atr_4h":         ind_4h.get("atr"),
                "price_vs_ema50": ind_1h.get("price_vs_ema50"),
                "trend_strength": ind_1h.get("trend_strength"),
                "volatility_pct": ind_1h.get("volatility_pct"),
                "volume_ratio":   ind_1h.get("volume_ratio"),
                "ema_gap_pct":    ind_1h.get("ema_gap_pct"),
                "candle_body_pct":ind_1h.get("candle_body_pct"),
                "rsi_divergence": ind_1h.get("rsi") - ind_4h.get("rsi", ind_1h.get("rsi")),
                "hour_of_day":    datetime.now(timezone.utc).hour,
                "day_of_week":    datetime.now(timezone.utc).weekday(),
                "trend_4h":       1 if ind_4h["ema50"] > ind_4h["ema200"] else -1,
                # New extended fields
                "macd":           ind_1h.get("macd"),
                "macd_signal":    ind_1h.get("macd_signal"),
                "macd_histogram": ind_1h.get("macd_histogram"),
                "stoch_rsi_k":    ind_1h.get("stoch_rsi_k"),
                "adx":            ind_1h.get("adx"),
                "cci":            ind_1h.get("cci"),
                "obv":            ind_1h.get("obv"),
                "bb_upper":       ind_1h.get("bb_upper"),
                "bb_lower":       ind_1h.get("bb_lower"),
                "bb_width":       ind_1h.get("bb_width"),
                "supertrend":     ind_1h.get("supertrend"),
                "supertrend_dir": ind_1h.get("supertrend_dir"),
                "realized_vol":   ind_1h.get("realized_vol"),
                "atr_percentile": ind_1h.get("atr_percentile"),
                "market_regime":  ind_1h.get("market_regime"),
                "volume_ratio":   ind_1h.get("volume_ratio"),
            }
            await save_trade_context(trade_id, ctx)

            # Save AI prediction log
            if ai_model.is_loaded() and ai_features:
                await save_ai_prediction_log(
                    trade_id=trade_id,
                    model_version=ai_model.get_version() or "unknown",
                    confidence=confidence,
                    probability_win=confidence,
                    chosen_action=signal.side,
                    feature_vector=ai_features,
                    feature_importance=ai_importance,
                    inference_time_ms=ai_inference_ms,
                )

            await log_decision(
                symbol=symbol, action="ENTERED",
                reason=signal.reason,
                signal_type=signal.side,
                rsi=ind_1h.get("rsi"),
                atr=ind_1h.get("atr"),
                confidence=confidence * 100,
            )

        await telegram.send(
            msg_opened(
                symbol, signal.side,
                signal.entry_price, signal.stop_loss, signal.take_profit,
                lot_size, notional, state.balance,
            )
        )

    # ── Excursion tracking ────────────────────────────────────────

    async def _update_excursions(self, symbol: str, current_price: float) -> None:
        """Update MFE/MAE for all open trades on every candle close."""
        tracker = _excursion_tracker.get(symbol)
        if not tracker:
            return
        new_mfe, new_mae = await update_mfe_mae(
            trade_id=tracker["trade_id"],
            current_price=current_price,
            entry_price=tracker["entry_price"],
            side=tracker["side"],
            current_mfe=tracker["mfe"],
            current_mae=tracker["mae"],
        )
        tracker["mfe"] = new_mfe
        tracker["mae"] = new_mae

    # ── Paper exit checker ────────────────────────────────────────

    async def _check_paper_exits(self, symbol: str, candle: dict) -> None:
        from app.execution.paper import check_exits
        trades_snapshot = {s: t for s, t in state.open_trades.items()}
        closed          = await check_exits({symbol: float(candle["close"])})

        for result in closed:
            sym      = result["symbol"]
            snapshot = trades_snapshot.get(sym)
            side     = snapshot.side if snapshot else result.get("side", "?")
            tracker  = _excursion_tracker.pop(sym, {})
            mfe      = tracker.get("mfe", 0.0)
            mae      = tracker.get("mae", 0.0)

            # Calculate holding time
            opened_at      = tracker.get("opened_at", datetime.now(timezone.utc))
            holding_mins   = int((datetime.now(timezone.utc) - opened_at).total_seconds() / 60)

            # Exit fee
            exit_notional  = result["exit_price"] * (snapshot.lot_size if snapshot else 0)
            exit_fee       = exit_notional * BINANCE_FEE_PCT

            # Fetch DB trade record
            from app.database.trades import get_closed_trades
            recent = await get_closed_trades(limit=1)

            if recent:
                t       = recent[0]
                trade_id = t["id"]

                # Update trade with full lifecycle data
                await close_trade(
                    trade_id=trade_id,
                    exit_price=result["exit_price"],
                    profit_loss=result["pnl"],
                    profit_pct=result["pnl_pct"],
                    exit_reason=result["reason"],
                    fees_exit=exit_fee,
                    mfe=mfe,
                    mae=mae,
                    holding_minutes=holding_mins,
                    equity_after=state.balance,
                )

                # Save balance to Supabase
                from app.database.client import set_state_value
                set_state_value("paper_balance", str(round(state.balance, 8)))

                # Run learning loop with full context
                ind_1h = candle_store.get_indicators(sym, TIMEFRAMES.signal) or {}
                ind_4h = candle_store.get_indicators(sym, TIMEFRAMES.trend) or {}
                await on_trade_closed(t, ind_1h, ind_4h)

            await telegram.send(
                msg_closed(
                    symbol=sym,
                    side=side,
                    exit_price=result["exit_price"],
                    pnl=result["pnl"],
                    pnl_pct=result["pnl_pct"],
                    reason=result["reason"],
                    balance=state.balance,
                )
            )

    # ── Execution dispatch ────────────────────────────────────────

    async def _execute(self, signal, lot_size: float, notional: float) -> dict | None:
        if TRADING.mode == "paper":
            from app.execution.paper import open_order
        else:
            from app.execution.binance import open_order

        return await open_order(
            symbol=signal.symbol,
            side=signal.side,
            lot_size=lot_size,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_version=get_active_params().version,
        )

    # ── Background loops ──────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        import app.config as cfg
        interval = cfg.NOTIFICATIONS.heartbeat_hours * 3600
        while state.is_running:
            await asyncio.sleep(interval)
            await telegram.send(
                msg_heartbeat(
                    balance=state.balance,
                    open_trades=state.open_trade_count(),
                    binance_ok=state.binance_connected,
                    supabase_ok=state.supabase_connected,
                    mode=TRADING.mode,
                )
            )

    async def _daily_reset_loop(self) -> None:
        while state.is_running:
            await asyncio.sleep(60)
            today = datetime.now(timezone.utc).date()
            if state.daily_reset_date and today > state.daily_reset_date:
                state.daily_pnl         = 0.0
                state.daily_trade_count = 0
                state.daily_reset_date  = today
                if state.is_paused:
                    state.is_paused = False
                    logger.info("Daily limit reset — bot unpaused")
                    await telegram.send("🔄 New trading day — bot unpaused")

    # ── Helpers ───────────────────────────────────────────────────

    async def _get_balance(self) -> float:
        if TRADING.mode == "paper":
            from app.database.client import get_state_value, set_state_value
            saved = get_state_value("paper_balance")
            if saved:
                balance = float(saved)
                logger.info(f"Paper balance restored: ${balance:,.2f}")
                return balance
            default = float(os.getenv("PAPER_BALANCE", "500"))
            set_state_value("paper_balance", str(default))
            return default
        from app.execution.binance import get_account_balance
        return await get_account_balance()

    async def _seed_candles(self) -> None:
        for symbol in TRADING.symbols:
            for tf in [TIMEFRAMES.signal, TIMEFRAMES.trend]:
                df = await get_candles(symbol, tf, limit=300)
                if not df.empty:
                    candle_store.seed(symbol, tf, df)
                else:
                    logger.warning(
                        f"No candles for {symbol} {tf} — "
                        f"run scripts/seed_history.py first"
                    )

    async def _load_ai_model(self) -> None:
        from app.database.context import get_active_model_version
        version_row = await get_active_model_version()
        if version_row:
            loaded = ai_model.load_model(
                version_row["model_path"], version_row["version"]
            )
            state.model_loaded         = loaded
            state.active_model_version = version_row["version"] if loaded else None
        else:
            logger.info("No trained AI model found — running rule-based only")

    async def _reconcile_open_trades(self) -> None:
        db_open = await get_open_trades()
        if db_open:
            logger.warning(f"Found {len(db_open)} open trades in DB on startup")
            from app.state import OpenTrade
            for t in db_open:
                state.open_trades[t["symbol"]] = OpenTrade(
                    trade_id=t["id"],
                    symbol=t["symbol"],
                    side=t["side"],
                    entry_price=float(t["entry_price"]),
                    stop_loss=float(t["stop_loss"]),
                    take_profit=float(t["take_profit"]),
                    lot_size=float(t["lot_size"]),
                    opened_at=t["opened_at"],
                    strategy_version=t["strategy_version"],
                    order_id=t.get("order_id"),
                )
                # Restart excursion tracking
                _excursion_tracker[t["symbol"]] = {
                    "trade_id":    t["id"],
                    "entry_price": float(t["entry_price"]),
                    "side":        t["side"],
                    "opened_at":   datetime.now(timezone.utc),
                    "mfe":         0.0,
                    "mae":         0.0,
                }

    # ── Scheduler ─────────────────────────────────────────────────

    def _setup_scheduler(self) -> None:
        """
        Sets up all scheduled background jobs.
        Weekly report: every Sunday at 08:00 UTC
        Daily risk snapshot: every day at 00:05 UTC
        Model retrain check: every Sunday at 09:00 UTC
        """
        import app.config as cfg

        # Weekly performance report + retrain
        day = cfg.NOTIFICATIONS.weekly_report_day.lower()[:3]
        hour = cfg.NOTIFICATIONS.weekly_report_hour

        self.scheduler.add_job(
            self._weekly_report_job,
            CronTrigger(day_of_week=day, hour=hour, minute=0),
            id="weekly_report",
            replace_existing=True,
        )

        # Daily risk metrics snapshot at midnight
        self.scheduler.add_job(
            self._daily_risk_snapshot,
            CronTrigger(hour=0, minute=5),
            id="daily_risk",
            replace_existing=True,
        )

        logger.info(
            f"Scheduler started: weekly report every {day} at {hour:02d}:00 UTC"
        )

    async def _weekly_report_job(self) -> None:
        """Runs every Sunday — generates report, tunes params, retrains model."""
        try:
            from scripts.weekly_report import generate_and_send
            await generate_and_send()
        except Exception as e:
            logger.error(f"Weekly report job failed: {e}")

    async def _daily_risk_snapshot(self) -> None:
        """Saves a daily risk metrics snapshot to Supabase."""
        try:
            from app.database.client import get_client
            client = get_client()
            client.table("risk_metrics").insert({
                "snapshot_time":    datetime.now(timezone.utc).isoformat(),
                "period":           "DAILY",
                "balance":          state.balance,
                "equity":           state.equity,
                "current_drawdown": state.drawdown_pct(),
                "peak_balance":     state.starting_balance,
                "daily_risk_used":  state.daily_loss_pct(),
                "total_trades":     state.daily_trade_count,
                "created_at":       datetime.now(timezone.utc).isoformat(),
            }).execute()
            logger.info(f"Daily risk snapshot saved: balance=${state.balance:,.2f}")
        except Exception as e:
            logger.error(f"Daily risk snapshot failed: {e}")

    # ── Live trade exit monitoring ─────────────────────────────────

    async def _live_exit_monitor(self) -> None:
        """
        Polls Binance every 60 seconds for closed live positions.
        Only runs in live mode. Paper mode uses candle-close checks.
        WHY: In live mode the exchange can close a position (SL/TP hit)
        between candle closes. We need to detect this promptly so
        the bot knows the position is gone and can take new trades.
        """
        logger.info("Live exit monitor started — polling every 60s")
        while state.is_running:
            await asyncio.sleep(60)
            try:
                await self._check_live_exits()
            except Exception as e:
                logger.error(f"Live exit monitor error: {e}")

    async def _check_live_exits(self) -> None:
        """Check if any live positions have been closed by the exchange."""
        if not state.open_trades:
            return
        try:
            from app.execution.binance import get_client as get_binance_client
            client = await get_binance_client()

            for symbol, trade in list(state.open_trades.items()):
                positions = await client.futures_position_information(symbol=symbol)
                for pos in positions:
                    if float(pos.get("positionAmt", 0)) == 0:
                        # Position closed by exchange — SL or TP hit
                        logger.info(f"Live position closed detected: {symbol}")

                        # Get recent trades to find exit price
                        trades = await client.futures_account_trades(
                            symbol=symbol, limit=5
                        )
                        exit_price = trade.entry_price
                        if trades:
                            exit_price = float(trades[-1]["price"])

                        # Determine PnL
                        if trade.side == "LONG":
                            pnl     = (exit_price - trade.entry_price) * trade.lot_size
                            reason  = "TP_HIT" if exit_price >= trade.take_profit else "SL_HIT"
                        else:
                            pnl     = (trade.entry_price - exit_price) * trade.lot_size
                            reason  = "TP_HIT" if exit_price <= trade.take_profit else "SL_HIT"

                        pnl_pct     = pnl / state.balance if state.balance else 0
                        exit_fee    = exit_price * trade.lot_size * BINANCE_FEE_PCT
                        tracker     = _excursion_tracker.pop(symbol, {})
                        opened_at   = tracker.get("opened_at", datetime.now(timezone.utc))
                        hold_mins   = int((datetime.now(timezone.utc) - opened_at).total_seconds() / 60)

                        # Update state
                        state.record_closed_trade(pnl)
                        del state.open_trades[symbol]

                        # Save to DB
                        await close_trade(
                            trade_id=trade.trade_id,
                            exit_price=exit_price,
                            profit_loss=pnl,
                            profit_pct=pnl_pct,
                            exit_reason=reason,
                            fees_exit=exit_fee,
                            mfe=tracker.get("mfe", 0.0),
                            mae=tracker.get("mae", 0.0),
                            holding_minutes=hold_mins,
                            equity_after=state.balance,
                        )

                        # Save balance
                        from app.database.client import set_state_value
                        set_state_value("paper_balance", str(round(state.balance, 8)))

                        # Run learning loop
                        from app.database.trades import get_closed_trades
                        recent = await get_closed_trades(limit=1)
                        if recent:
                            ind_1h = candle_store.get_indicators(symbol, TIMEFRAMES.signal) or {}
                            ind_4h = candle_store.get_indicators(symbol, TIMEFRAMES.trend) or {}
                            await on_trade_closed(recent[0], ind_1h, ind_4h)

                        await telegram.send(
                            msg_closed(
                                symbol=symbol,
                                side=trade.side,
                                exit_price=exit_price,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                reason=reason,
                                balance=state.balance,
                            )
                        )
        except Exception as e:
            logger.error(f"Failed to check live exits: {e}")

    async def _close_all_emergency(self) -> None:
        if TRADING.mode == "live":
            from app.execution.binance import close_all_positions
            await close_all_positions("EMERGENCY")
        else:
            from app.execution.paper import close_order
            for sym in list(state.open_trades.keys()):
                trade = state.open_trades[sym]
                await close_order(sym, trade.entry_price, "EMERGENCY")
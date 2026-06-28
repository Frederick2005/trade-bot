import asyncio
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
from app.database.trades import save_trade, close_trade, get_open_trades, count_closed_trades
from app.database.market import get_candles
from app.database.context import save_trade_context, log_decision, log_bot_event
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


class TradingEngine:
    def __init__(self):
        self.strategy    = EmaRsiStrategy(version="v1.0")
        self._last_signal: dict[str, str] = {}   # symbol -> last side signalled

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        log_config()
        logger.info(f"Starting engine in {TRADING.mode.upper()} mode")

        # Check DB connection
        state.supabase_connected = check_connection()
        if not state.supabase_connected:
            logger.critical("Cannot connect to Supabase — aborting")
            return

        # Seed candle buffers from Supabase history
        await self._seed_candles()

        # Load AI model if one exists
        await self._load_ai_model()

        # Sync open trades from broker on restart
        await self._reconcile_open_trades()

        # Restore balance from last session (or initialise if first run)
        from app.database.client import get_state_value, set_state_value
        restored               = await self._get_balance()
        state.balance          = restored
        state.equity           = restored

        # Restore starting_balance for accurate drawdown calculation across restarts
        saved_start = get_state_value('starting_balance')
        if saved_start:
            state.starting_balance = float(saved_start)
        else:
            state.starting_balance = restored
            set_state_value('starting_balance', str(restored))

        state.is_running       = True
        state.daily_reset_date = datetime.now(timezone.utc).date()

        logger.info(
            f"Balance restored: ${state.balance:,.2f} | "
            f"All-time start: ${state.starting_balance:,.2f} | "
            f"Drawdown: {state.drawdown_pct():.2%}"
        )

        # Start background jobs
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._daily_reset_loop())

        # Start Telegram listener (non-blocking)
        asyncio.create_task(telegram.start_listener())

        await telegram.send(
            f"🚀 *Bot started*\n"
            f"Mode: `{TRADING.mode.upper()}`\n"
            f"Balance: `${state.balance:,.2f}`\n"
            f"Symbols: `{', '.join(TRADING.symbols)}`"
        )

        # Start Binance streams — blocks here until stopped
        await start_streams(on_candle_close=self._on_candle_close)

    # ── Candle close handler ──────────────────────────────────────────────────

    async def _on_candle_close(
        self, symbol: str, timeframe: str, candle: dict
    ) -> None:
        """Called by the stream on every closed candle."""

        # Only process signal timeframe (1H) — 4H is just for confluence
        if timeframe != TIMEFRAMES.signal:
            return

        # Check emergency stop before doing anything
        stop, reason = check_emergency_stop()
        if stop:
            logger.critical(reason)
            await telegram.send(drawdown_stop(state.drawdown_pct(), state.balance))
            await self._close_all_emergency()
            state.is_paused = True
            return

        # Check daily loss limit
        if state.daily_loss_pct() >= TRADING.daily_loss_limit:
            if not state.is_paused:
                state.is_paused = True
                await telegram.send(
                    daily_loss_limit_hit(state.daily_loss_pct(), state.balance)
                )
            return

        # Check paper exit conditions
        if TRADING.mode == "paper":
            await self._check_paper_exits(symbol, candle)

        # Evaluate strategy
        await self._evaluate_signal(symbol)

    # ── Signal evaluation ─────────────────────────────────────────────────────

    async def _evaluate_signal(self, symbol: str) -> None:
        ind_1h = candle_store.get_indicators(symbol, TIMEFRAMES.signal)
        ind_4h = candle_store.get_indicators(symbol, TIMEFRAMES.trend)

        if ind_1h is None or ind_4h is None:
            return

        signal = self.strategy.evaluate(symbol, ind_1h, ind_4h)

        if signal is None:
            await log_decision(
                symbol=symbol,
                action="SKIPPED",
                reason="No signal from strategy",
                rsi=ind_1h.get("rsi"),
                atr=ind_1h.get("atr"),
            )
            return

        # ── AI confidence filter ───────────────────────────────────
        confidence = signal.confidence
        ai_override = False

        if ai_model.is_loaded():
            features   = build_feature_vector(
                ind_1h, ind_4h,
                candle_time=datetime.now(timezone.utc),
                recent_win_rate=0.5,
                current_drawdown=state.drawdown_pct(),
            )
            ai_conf, ai_sig = ai_model.predict(features)
            confidence  = ai_conf

            if ai_conf < AI.min_confidence:
                await log_decision(
                    symbol=symbol,
                    action="SKIPPED",
                    reason=f"AI confidence {ai_conf:.2%} below threshold {AI.min_confidence:.2%}",
                    signal_type=signal.side,
                    rsi=ind_1h.get("rsi"),
                    confidence=ai_conf * 100,
                )
                return
            ai_override = True

        # ── Risk guards ────────────────────────────────────────────
        allowed, reason = check_all(symbol, signal.side)
        if not allowed:
            await log_decision(
                symbol=symbol,
                action="BLOCKED",
                reason=reason,
                signal_type=signal.side,
                rsi=ind_1h.get("rsi"),
                confidence=confidence * 100,
            )
            return

        # ── Position sizing ────────────────────────────────────────
        leverage = auto_reduce_leverage(state.balance, TRADING.max_leverage)
        lot_size, notional = calculate_lot_size(
            balance=state.balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            leverage=leverage,
        )
        if lot_size == 0:
            await log_decision(symbol, "BLOCKED", "Position size too small")
            return

        # ── Execute ────────────────────────────────────────────────
        trade_result = await self._execute(signal, lot_size, notional)
        if trade_result is None:
            return

        # ── Save to DB ─────────────────────────────────────────────
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
        )

        if trade_id:
            # Save indicator context for learning
            ctx = {
                **signal.indicators,
                "ema50_4h":    ind_4h.get("ema50"),
                "ema200_4h":   ind_4h.get("ema200"),
                "rsi_4h":      ind_4h.get("rsi"),
                "atr_4h":      ind_4h.get("atr"),
                "hour_of_day": datetime.now(timezone.utc).hour,
                "day_of_week": datetime.now(timezone.utc).weekday(),
                "trend_4h":    1 if ind_4h["ema50"] > ind_4h["ema200"] else -1,
            }
            await save_trade_context(trade_id, ctx)
            await log_decision(
                symbol=symbol,
                action="ENTERED",
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

    # ── Execution dispatch ────────────────────────────────────────────────────

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

    # ── Paper exit checker ────────────────────────────────────────────────────

    async def _check_paper_exits(self, symbol: str, candle: dict) -> None:
        from app.execution.paper import check_exits

        # Snapshot open trades BEFORE check_exits removes them
        trades_snapshot = {s: t for s, t in state.open_trades.items()}

        closed = await check_exits({symbol: float(candle["close"])})

        for result in closed:
            sym  = result["symbol"]

            # Get side from snapshot captured before trade was removed
            snapshot = trades_snapshot.get(sym)
            side     = snapshot.side if snapshot else result.get("side", "?")

            # Fetch full trade record for learning loop
            from app.database.trades import get_closed_trades
            recent = await get_closed_trades(limit=1)
            if recent:
                t      = recent[0]
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

    # ── Background loops ──────────────────────────────────────────────────────

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
        """Resets daily P&L at midnight UTC and unpauses if paused by daily limit."""
        while state.is_running:
            await asyncio.sleep(60)
            today = datetime.now(timezone.utc).date()
            if state.daily_reset_date and today > state.daily_reset_date:
                state.daily_pnl        = 0.0
                state.daily_trade_count = 0
                state.daily_reset_date = today
                if state.is_paused:
                    state.is_paused = False
                    logger.info("Daily limit reset — bot unpaused")
                    await telegram.send("🔄 New trading day — bot unpaused")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_balance(self) -> float:
        if TRADING.mode == "paper":
            import os
            from app.database.client import get_state_value, set_state_value

            # Try to restore saved balance from Supabase first
            saved = get_state_value("paper_balance")
            if saved:
                balance = float(saved)
                logger.info(f"Paper balance restored from last session: ${balance:,.2f}")
                return balance

            # First ever run — use env default and save it
            default = float(os.getenv("PAPER_BALANCE", "500"))
            set_state_value("paper_balance", str(default))
            logger.info(f"Paper balance initialised for first time: ${default:,.2f}")
            return default

        # Live mode — fetch real balance from Binance
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
                        f"No historical candles found for {symbol} {tf} — "
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
            logger.warning(
                f"Found {len(db_open)} open trades in DB on startup — "
                f"review manually or they will be tracked in state"
            )
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

    async def _close_all_emergency(self) -> None:
        if TRADING.mode == "live":
            from app.execution.binance import close_all_positions
            await close_all_positions("EMERGENCY")
        else:
            from app.execution.paper import close_order
            for sym in list(state.open_trades.keys()):
                trade = state.open_trades[sym]
                await close_order(sym, trade.entry_price, "EMERGENCY")
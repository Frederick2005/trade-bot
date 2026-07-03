"""
TradingEngine — updated to support:
  - Up to 10 concurrent trades across all symbols
  - 15m signal + 1h trend timeframes
  - Trailing stop + breakeven logic
  - Position tracking by trade_id (not just symbol)
"""
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

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_OPEN_TRADES    = 10     # maximum concurrent positions across all symbols
BREAKEVEN_ATR_MULT = 1.0    # move SL to breakeven after price moves +1×ATR
TRAILING_ATR_MULT  = 1.5    # trail SL at 1.5×ATR behind price after breakeven


class ActiveTrade:
    """Tracks a single open position with trailing stop state."""
    def __init__(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        atr: float,
        opened_at: str,
        strategy_version: str,
        order_id: str | None = None,
    ):
        self.trade_id         = trade_id
        self.symbol           = symbol
        self.side             = side
        self.entry_price      = entry_price
        self.stop_loss        = stop_loss
        self.take_profit      = take_profit
        self.lot_size         = lot_size
        self.atr              = atr
        self.opened_at        = opened_at
        self.strategy_version = strategy_version
        self.order_id         = order_id
        self.breakeven_hit    = False   # has SL been moved to breakeven yet?
        self.highest_price    = entry_price  # for trailing stop tracking


class TradingEngine:
    def __init__(self):
        self.strategy      = EmaRsiStrategy(version="v1.0")
        self.active_trades: dict[str, ActiveTrade] = {}  # trade_id -> ActiveTrade
        self._last_signal:  dict[str, str]         = {}  # symbol -> last side

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        log_config()
        logger.info(f"Starting engine in {TRADING.mode.upper()} mode")
        logger.info(f"Max concurrent trades: {MAX_OPEN_TRADES}")

        state.supabase_connected = check_connection()
        if not state.supabase_connected:
            logger.critical("Cannot connect to Supabase — aborting")
            return

        await self._seed_candles()
        await self._load_ai_model()
        await self._reconcile_open_trades()

        from app.database.client import get_state_value, set_state_value
        restored               = await self._get_balance()
        state.balance          = restored
        state.equity           = restored

        saved_start = get_state_value("starting_balance")
        if saved_start:
            state.starting_balance = float(saved_start)
        else:
            state.starting_balance = restored
            set_state_value("starting_balance", str(restored))

        state.is_running       = True
        state.daily_reset_date = datetime.now(timezone.utc).date()

        logger.info(
            f"Balance: ${state.balance:,.2f} | "
            f"Start: ${state.starting_balance:,.2f} | "
            f"Drawdown: {state.drawdown_pct():.2%}"
        )

        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._daily_reset_loop())
        asyncio.create_task(telegram.start_listener())

        import os, time, tempfile
        last_start_file = os.path.join(tempfile.gettempdir(), "bot_last_start")
        now = time.time()
        send_startup_msg = True
        if os.path.exists(last_start_file):
            try:
                with open(last_start_file) as f:
                    last = float(f.read())
                if now - last < 600:
                    send_startup_msg = False
            except Exception:
                pass
        try:
            with open(last_start_file, "w") as f:
                f.write(str(now))
        except Exception:
            pass

        if send_startup_msg:
            await telegram.send(
                f"🚀 *Bot started*\n"
                f"Mode: `{TRADING.mode.upper()}`\n"
                f"Balance: `${state.balance:,.2f}`\n"
                f"Symbols: `{', '.join(TRADING.symbols)}`\n"
                f"Max trades: `{MAX_OPEN_TRADES}`\n"
                f"Timeframes: `{TIMEFRAMES.signal}/{TIMEFRAMES.trend}`"
            )

        await start_streams(on_candle_close=self._on_candle_close)

    # ── Candle close handler ──────────────────────────────────────────────────

    async def _on_candle_close(
        self, symbol: str, timeframe: str, candle: dict
    ) -> None:
        if timeframe != TIMEFRAMES.signal:
            return

        stop, reason = check_emergency_stop()
        if stop:
            logger.critical(reason)
            await telegram.send(drawdown_stop(state.drawdown_pct(), state.balance))
            await self._close_all_emergency()
            state.is_paused = True
            return

        if state.daily_loss_pct() >= TRADING.daily_loss_limit:
            if not state.is_paused:
                state.is_paused = True
                await telegram.send(
                    daily_loss_limit_hit(state.daily_loss_pct(), state.balance)
                )
            return

        # Update trailing stops for all open trades on this symbol
        current_price = float(candle["close"])
        await self._update_trailing_stops(symbol, current_price)

        # Check paper exits
        if TRADING.mode == "paper":
            await self._check_paper_exits(symbol, candle)

        # Evaluate new signal
        await self._evaluate_signal(symbol)

    # ── Trailing stop + breakeven logic ───────────────────────────────────────

    async def _update_trailing_stops(
        self, symbol: str, current_price: float
    ) -> None:
        """
        For each open trade on this symbol:
        1. Once profit >= 1×ATR → move SL to breakeven (entry price)
        2. After breakeven → trail SL at 1.5×ATR below highest price seen
        """
        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol != symbol:
                continue
            if trade.side != "LONG":
                continue

            profit_dist = current_price - trade.entry_price

            # Step 1 — breakeven
            if not trade.breakeven_hit:
                if profit_dist >= trade.atr * BREAKEVEN_ATR_MULT:
                    new_sl = trade.entry_price  # move to entry
                    if new_sl > trade.stop_loss:
                        trade.stop_loss    = new_sl
                        trade.breakeven_hit = True
                        logger.info(
                            f"Breakeven triggered: {symbol} trade={trade_id[:8]} "
                            f"SL moved to {new_sl:.2f}"
                        )

            # Step 2 — trailing stop
            if trade.breakeven_hit:
                if current_price > trade.highest_price:
                    trade.highest_price = current_price

                trailing_sl = trade.highest_price - (trade.atr * TRAILING_ATR_MULT)
                if trailing_sl > trade.stop_loss:
                    trade.stop_loss = trailing_sl
                    logger.debug(
                        f"Trailing stop updated: {symbol} "
                        f"trade={trade_id[:8]} SL={trailing_sl:.2f}"
                    )

    # ── Signal evaluation ─────────────────────────────────────────────────────

    async def _evaluate_signal(self, symbol: str) -> None:
        # Check global trade cap
        if len(self.active_trades) >= MAX_OPEN_TRADES:
            logger.debug(f"Max trades ({MAX_OPEN_TRADES}) reached — skipping {symbol}")
            return

        ind_signal = candle_store.get_indicators(symbol, TIMEFRAMES.signal)
        ind_trend  = candle_store.get_indicators(symbol, TIMEFRAMES.trend)

        if ind_signal is None or ind_trend is None:
            return

        # Strategy expects (symbol, ind_signal, ind_trend)
        # ema_rsi.py uses ind_4h as signal and ind_1h as trend
        # so pass (symbol, ind_trend, ind_signal) to match parameter names
        signal = self.strategy.evaluate(symbol, ind_trend, ind_signal)

        if signal is None:
            await log_decision(
                symbol=symbol,
                action="SKIPPED",
                reason="No signal",
                rsi=ind_signal.get("rsi"),
                atr=ind_signal.get("atr"),
            )
            return

        # ── AI confidence filter ───────────────────────────────────
        confidence  = signal.confidence
        if ai_model.is_loaded():
            features = build_feature_vector(
                ind_trend, ind_signal,
                candle_time=datetime.now(timezone.utc),
                recent_win_rate=0.5,
                current_drawdown=state.drawdown_pct(),
            )
            ai_conf, _ = ai_model.predict(features)
            confidence  = ai_conf
            if ai_conf < AI.min_confidence:
                await log_decision(
                    symbol=symbol,
                    action="SKIPPED",
                    reason=f"AI conf {ai_conf:.2%} < {AI.min_confidence:.2%}",
                    signal_type=signal.side,
                    rsi=ind_signal.get("rsi"),
                    confidence=ai_conf * 100,
                )
                return

        # ── Risk guards ────────────────────────────────────────────
        allowed, reason = check_all(symbol, signal.side)
        if not allowed:
            await log_decision(
                symbol=symbol, action="BLOCKED", reason=reason,
                signal_type=signal.side, rsi=ind_signal.get("rsi"),
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
            await log_decision(symbol, "BLOCKED", "Position too small")
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
            # Track in active trades with ATR for trailing stop
            atr = ind_signal.get("atr", 0.0)
            self.active_trades[trade_id] = ActiveTrade(
                trade_id=trade_id,
                symbol=symbol,
                side=signal.side,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                lot_size=lot_size,
                atr=atr,
                opened_at=datetime.now(timezone.utc).isoformat(),
                strategy_version=get_active_params().version,
                order_id=trade_result.get("order_id"),
            )

            ctx = {
                **signal.indicators,
                "hour_of_day": datetime.now(timezone.utc).hour,
                "day_of_week": datetime.now(timezone.utc).weekday(),
            }
            await save_trade_context(trade_id, ctx)
            await log_decision(
                symbol=symbol, action="ENTERED", reason=signal.reason,
                signal_type=signal.side, rsi=ind_signal.get("rsi"),
                atr=ind_signal.get("atr"), confidence=confidence * 100,
            )

            logger.info(
                f"Trade opened: {symbol} {signal.side} | "
                f"entry={signal.entry_price:.2f} SL={signal.stop_loss:.2f} "
                f"TP={signal.take_profit:.2f} | "
                f"Active trades: {len(self.active_trades)}/{MAX_OPEN_TRADES}"
            )

        await telegram.send(
            msg_opened(
                symbol, signal.side,
                signal.entry_price, signal.stop_loss, signal.take_profit,
                lot_size, notional, state.balance,
            )
        )

    # ── Execution ─────────────────────────────────────────────────────────────

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

        # Build price map for all symbols
        price_map = {symbol: float(candle["close"])}

        # Check exits using updated SL/TP from trailing stop logic
        # Pass current active trades so paper executor uses updated SL values
        closed = await check_exits(price_map)

        for result in closed:
            sym      = result["symbol"]
            trade_id = result.get("trade_id")

            # Get side from active trades
            trade = None
            if trade_id and trade_id in self.active_trades:
                trade = self.active_trades.pop(trade_id)
            else:
                # Fallback: find by symbol
                for tid, t in list(self.active_trades.items()):
                    if t.symbol == sym:
                        trade = self.active_trades.pop(tid)
                        break

            side = trade.side if trade else result.get("side", "LONG")

            from app.database.trades import get_closed_trades
            recent = await get_closed_trades(limit=1)
            if recent:
                t      = recent[0]
                ind_s  = candle_store.get_indicators(sym, TIMEFRAMES.signal) or {}
                ind_t  = candle_store.get_indicators(sym, TIMEFRAMES.trend)  or {}
                await on_trade_closed(t, ind_s, ind_t)

            logger.info(
                f"Trade closed: {sym} {side} | "
                f"exit={result['exit_price']:.2f} "
                f"pnl={result['pnl']:+.2f} ({result['reason']}) | "
                f"Active trades: {len(self.active_trades)}/{MAX_OPEN_TRADES}"
            )

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
                    open_trades=len(self.active_trades),
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_balance(self) -> float:
        if TRADING.mode == "paper":
            import os
            from app.database.client import get_state_value, set_state_value
            saved = get_state_value("paper_balance")
            if saved:
                balance = float(saved)
                logger.info(f"Paper balance restored: ${balance:,.2f}")
                return balance
            default = float(os.getenv("PAPER_BALANCE", "500"))
            set_state_value("paper_balance", str(default))
            logger.info(f"Paper balance initialised: ${default:,.2f}")
            return default
        from app.execution.binance import get_account_balance
        return await get_account_balance()

    async def _seed_candles(self) -> None:
        for symbol in TRADING.symbols:
            for tf in [TIMEFRAMES.signal, TIMEFRAMES.trend]:
                df = await get_candles(symbol, tf, limit=500)
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
            logger.info("No AI model found — running rule-based only")

    async def _reconcile_open_trades(self) -> None:
        db_open = await get_open_trades()
        if db_open:
            logger.warning(
                f"Found {len(db_open)} open trades in DB on startup"
            )
            for t in db_open:
                trade_id = t["id"]
                atr      = float(t.get("atr", 0.0) or 0.0)
                self.active_trades[trade_id] = ActiveTrade(
                    trade_id=trade_id,
                    symbol=t["symbol"],
                    side=t["side"],
                    entry_price=float(t["entry_price"]),
                    stop_loss=float(t["stop_loss"]),
                    take_profit=float(t["take_profit"]),
                    lot_size=float(t["lot_size"]),
                    atr=atr,
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
            for trade_id, trade in list(self.active_trades.items()):
                await close_order(trade.symbol, trade.entry_price, "EMERGENCY")
        self.active_trades.clear()
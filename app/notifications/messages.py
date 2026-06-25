from datetime import datetime


def signal_alert(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    confidence: float,
    reason: str,
) -> str:
    icon = "🟢" if side == "LONG" else "🔴"
    rr   = abs(tp - entry) / abs(entry - sl) if entry != sl else 0
    return (
        f"{icon} *SIGNAL: {symbol} {side}*\n"
        f"Entry:  `${entry:,.2f}`\n"
        f"SL:     `${sl:,.2f}`  ({(sl - entry) / entry * 100:+.2f}%)\n"
        f"TP:     `${tp:,.2f}`  ({(tp - entry) / entry * 100:+.2f}%)\n"
        f"R:R:    `1:{rr:.1f}`\n"
        f"Confidence: `{confidence * 100:.0f}%`\n"
        f"Reason: _{reason}_"
    )


def trade_opened(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    lot_size: float,
    notional: float,
    balance: float,
) -> str:
    icon = "📈" if side == "LONG" else "📉"
    return (
        f"{icon} *TRADE OPENED: {symbol}*\n"
        f"Side:     `{side}`\n"
        f"Entry:    `${entry:,.2f}`\n"
        f"SL:       `${sl:,.2f}`\n"
        f"TP:       `${tp:,.2f}`\n"
        f"Size:     `{lot_size:.6f}` (${notional:,.2f} notional)\n"
        f"Balance:  `${balance:,.2f}`"
    )


def trade_closed(
    symbol: str,
    side: str,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    reason: str,
    balance: float,
) -> str:
    icon = "✅" if pnl >= 0 else "❌"
    return (
        f"{icon} *TRADE CLOSED: {symbol}*\n"
        f"Side:    `{side}`\n"
        f"Exit:    `${exit_price:,.2f}`\n"
        f"P&L:     `{pnl:+.4f} USDT` (`{pnl_pct:+.2%}`)\n"
        f"Reason:  `{reason}`\n"
        f"Balance: `${balance:,.2f}`"
    )


def daily_loss_limit_hit(loss_pct: float, balance: float) -> str:
    return (
        f"⚠️ *DAILY LOSS LIMIT HIT*\n"
        f"Loss today: `{loss_pct:.2%}`\n"
        f"Bot paused until midnight UTC\n"
        f"Balance: `${balance:,.2f}`"
    )


def drawdown_stop(drawdown_pct: float, balance: float) -> str:
    return (
        f"🚨 *MAX DRAWDOWN REACHED — BOT STOPPED*\n"
        f"Drawdown: `{drawdown_pct:.2%}`\n"
        f"Balance: `${balance:,.2f}`\n"
        f"All positions closed. Manual review required."
    )


def weekly_report(
    period: str,
    total_trades: int,
    winning: int,
    win_rate: float,
    total_pnl: float,
    best_trade: float,
    worst_trade: float,
    balance: float,
    model_version: str | None,
) -> str:
    return (
        f"📊 *Weekly Report — {period}*\n\n"
        f"Trades:     `{total_trades}`\n"
        f"Win rate:   `{win_rate:.1%}`  ({winning}W / {total_trades - winning}L)\n"
        f"Total P&L:  `{total_pnl:+.4f} USDT`\n"
        f"Best trade: `{best_trade:+.4f} USDT`\n"
        f"Worst:      `{worst_trade:+.4f} USDT`\n"
        f"Balance:    `${balance:,.2f}`\n"
        f"AI model:   `{model_version or 'none'}`"
    )


def heartbeat(
    balance: float,
    open_trades: int,
    binance_ok: bool,
    supabase_ok: bool,
    mode: str,
) -> str:
    b_icon = "🟢" if binance_ok else "🔴"
    s_icon = "🟢" if supabase_ok else "🔴"
    return (
        f"💓 *Heartbeat*  `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
        f"Balance:    `${balance:,.2f}`\n"
        f"Open trades: `{open_trades}`\n"
        f"Binance:    {b_icon}  Supabase: {s_icon}\n"
        f"Mode:       `{mode.upper()}`"
    )


def bot_paused(by: str = "command") -> str:
    return f"⏸ *Bot paused* — triggered by `{by}`\nNo new trades will be opened."


def bot_resumed() -> str:
    return "▶️ *Bot resumed* — accepting new signals."
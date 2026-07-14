"""
app/database/trades.py
Trade CRUD operations with full lifecycle fields.

WHY EXTENDED TRADE DATA MAKES THE AI SMARTER:
──────────────────────────────────────────────
fees_paid
  Every trade costs money in fees. AI learns that
  a signal needs a minimum expected move to be worth
  taking after fees. Prevents taking marginal trades
  that look profitable but lose after costs.

r_multiple
  Expresses every trade result as a multiple of risk.
  +2R means won 2x what was risked. -1R means lost
  the full stop. AI learns to target high R-multiple
  setups and avoid low R ones. Standardises learning
  across different position sizes.

max_favourable_excursion (MFE)
  The furthest the trade went in your favour before
  closing. If MFE was +3% but you only captured +1%
  the AI learns your exit is too early. Over hundreds
  of trades this reveals exactly where TP should be.

max_adverse_excursion (MAE)
  The furthest the trade went against you before
  recovering. If MAE was -0.8% but stop was at -1.5%
  the AI learns stops are too wide. If MAE was -1.4%
  and stop at -1.5% — stops are too tight.
  MAE/MFE together are the most powerful stop/target
  optimisation data available.

holding_time_minutes
  How long winners vs losers stay open. If winners
  average 4 hours and losers average 45 minutes the
  AI learns early exits are usually losses and to
  hold winning trades longer.

equity_before / equity_after
  Tracks account growth trade by trade. AI learns
  how drawdown periods affect signal quality.

net_profit
  Profit after fees and funding costs. True profitability
  measure the AI optimises toward.

leverage_used
  AI learns if higher leverage trades perform differently
  than lower leverage ones — important for position sizing.
"""
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from app.database.client import get_client


async def save_trade(
    symbol: str,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    lot_size: float,
    strategy_version: str,
    account_balance: float,
    order_id: Optional[str] = None,
    leverage: int = 1,
    risk_amount: float = 0.0,
    fees_entry: float = 0.0,
) -> Optional[str]:
    try:
        reward_amount = abs(take_profit - entry_price) * lot_size
        risk_amount_  = risk_amount or abs(entry_price - stop_loss) * lot_size

        client = get_client()
        result = client.table("trades").insert({
            "symbol":           symbol,
            "side":             side,
            "entry_price":      entry_price,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "lot_size":         lot_size,
            "strategy_version": strategy_version,
            "account_balance":  account_balance,
            "order_id":         order_id,
            "status":           "OPEN",
            "opened_at":        datetime.now(timezone.utc).isoformat(),
            # Extended lifecycle fields
            "leverage_used":    leverage,
            "risk_amount":      round(risk_amount_, 6),
            "reward_amount":    round(reward_amount, 6),
            "fees_paid":        round(fees_entry, 6),
            "equity_before":    account_balance,
            "margin_used":      round(entry_price * lot_size / leverage, 2) if leverage > 0 else 0,
        }).execute()

        trade_id = result.data[0]["id"]
        logger.info(f"Trade saved: {symbol} {side} @ {entry_price} | id={trade_id}")
        return trade_id
    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        return None


async def close_trade(
    trade_id: str,
    exit_price: float,
    profit_loss: float,
    profit_pct: float,
    exit_reason: str,
    fees_exit: float = 0.0,
    mfe: float = 0.0,
    mae: float = 0.0,
    holding_minutes: int = 0,
    equity_after: float = 0.0,
) -> bool:
    """
    Close a trade with full lifecycle data.

    mfe = max_favourable_excursion — furthest price went in your favour
    mae = max_adverse_excursion    — furthest price went against you
    These two fields are gold for stop/target optimisation.
    """
    try:
        # Fetch the trade to calculate R-multiple
        client  = get_client()
        trade_r = client.table("trades").select(
            "entry_price,stop_loss,risk_amount,fees_paid,equity_before"
        ).eq("id", trade_id).limit(1).execute()

        r_multiple    = 0.0
        net_profit    = profit_loss - fees_exit
        total_fees    = fees_exit

        if trade_r.data:
            t          = trade_r.data[0]
            risk       = float(t.get("risk_amount") or 0)
            prev_fees  = float(t.get("fees_paid") or 0)
            total_fees = prev_fees + fees_exit
            net_profit = profit_loss - total_fees

            # R-multiple: how many units of risk did we win/lose
            if risk > 0:
                r_multiple = round(profit_loss / risk, 3)

        client.table("trades").update({
            "exit_price":              exit_price,
            "profit_loss":             round(profit_loss, 6),
            "profit_pct":              round(profit_pct, 6),
            "exit_reason":             exit_reason,
            "status":                  "CLOSED",
            "closed_at":               datetime.now(timezone.utc).isoformat(),
            # Extended lifecycle
            "fees_paid":               round(total_fees, 6),
            "net_profit":              round(net_profit, 6),
            "r_multiple":              r_multiple,
            "max_favourable_excursion":round(mfe, 6),
            "max_adverse_excursion":   round(mae, 6),
            "holding_time_minutes":    holding_minutes,
            "equity_after":            round(equity_after, 2),
        }).eq("id", trade_id).execute()

        logger.info(
            f"Trade closed: id={trade_id} | pnl={profit_loss:+.4f} | "
            f"R={r_multiple:+.2f} | MFE={mfe:.4f} | MAE={mae:.4f} | "
            f"held={holding_minutes}min | reason={exit_reason}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to close trade {trade_id}: {e}")
        return False


async def get_open_trades() -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("*")
            .eq("status", "OPEN")
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch open trades: {e}")
        return []


async def get_closed_trades(limit: int = 100) -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("*")
            .eq("status", "CLOSED")
            .order("closed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch closed trades: {e}")
        return []


async def get_trades_since(since: datetime) -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("*")
            .eq("status", "CLOSED")
            .gte("closed_at", since.isoformat())
            .order("closed_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch trades since {since}: {e}")
        return []


async def count_closed_trades() -> int:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("id", count="exact")
            .eq("status", "CLOSED")
            .execute()
        )
        return result.count or 0
    except Exception as e:
        logger.error(f"Failed to count closed trades: {e}")
        return 0


async def update_mfe_mae(
    trade_id: str,
    current_price: float,
    entry_price: float,
    side: str,
    current_mfe: float,
    current_mae: float,
) -> tuple[float, float]:
    """
    Update MFE and MAE tracking on every candle close.
    Returns updated (mfe, mae).

    WHY THIS MATTERS:
    Tracking the price excursion on every candle gives the AI
    a complete picture of how the trade breathed before resolution.
    Over hundreds of trades this reveals the optimal stop distance
    and take profit target for each market regime.
    """
    if side == "LONG":
        move_favour  = current_price - entry_price
        move_against = entry_price - current_price
    else:
        move_favour  = entry_price - current_price
        move_against = current_price - entry_price

    new_mfe = max(current_mfe, move_favour)
    new_mae = max(current_mae, move_against)
    return new_mfe, new_mae
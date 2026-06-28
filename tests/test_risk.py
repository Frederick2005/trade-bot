import pytest
from app.risk.sizing import calculate_lot_size
from app.risk.limits import enforce_leverage, check_min_balance
from app.risk.guards import check_all
from app.state import state, BotState


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before every test."""
    state.balance          = 500.0
    state.starting_balance = 500.0
    state.daily_pnl        = 0.0
    state.is_paused        = False
    state.open_trades      = {}
    yield


class TestPositionSizing:
    def test_basic_sizing(self):
        lot, notional = calculate_lot_size(500, 100_000, 98_500)
        assert lot > 0
        assert notional >= 5.0

    def test_below_minimum_notional(self):
        # Tiny balance, huge price, big SL → notional < $5
        lot, notional = calculate_lot_size(10, 100_000, 50_000)
        assert lot == 0.0
        assert notional == 0.0

    def test_leverage_cap(self):
        lot, notional = calculate_lot_size(500, 100_000, 99_000, leverage=5)
        max_notional = 500 * 5
        assert notional <= max_notional + 0.01

    def test_zero_stop_distance(self):
        lot, notional = calculate_lot_size(500, 100_000, 100_000)
        assert lot == 0.0

    def test_risk_percentage(self):
        # 1% of $500 = $5 risk, SL = $1000 away
        lot, _ = calculate_lot_size(500, 50_000, 49_000)
        expected_risk = 500 * 0.01
        actual_risk   = lot * 1_000
        assert abs(actual_risk - expected_risk) < 0.01


class TestLeverageLimits:
    def test_enforce_leverage_ok(self):
        assert enforce_leverage(3) == 3

    def test_enforce_leverage_capped(self):
        assert enforce_leverage(20) == 5

    def test_min_balance_ok(self):
        ok, _ = check_min_balance(200.0)
        assert ok is True

    def test_min_balance_fail(self):
        ok, reason = check_min_balance(30.0)
        assert ok is False
        assert "minimum" in reason


class TestRiskGuards:
    def test_paused_blocks_all(self):
        state.is_paused = True
        ok, reason = check_all("BTCUSDT", "LONG")
        assert ok is False
        assert "paused" in reason.lower()

    def test_max_open_trades(self):
        from app.state import OpenTrade
        for i in range(2):
            state.open_trades[f"SYM{i}"] = OpenTrade(
                trade_id=f"{i}", symbol=f"SYM{i}", side="LONG",
                entry_price=100, stop_loss=90, take_profit=120,
                lot_size=0.1, opened_at="", strategy_version="v1.0"
            )
        ok, reason = check_all("BTCUSDT", "LONG")
        assert ok is False
        assert "Max open trades" in reason

    def test_duplicate_symbol_blocked(self):
        from app.state import OpenTrade
        state.open_trades["BTCUSDT"] = OpenTrade(
            trade_id="1", symbol="BTCUSDT", side="LONG",
            entry_price=100, stop_loss=90, take_profit=120,
            lot_size=0.1, opened_at="", strategy_version="v1.0"
        )
        ok, reason = check_all("BTCUSDT", "LONG")
        assert ok is False
        assert "BTCUSDT" in reason

    def test_daily_loss_limit(self):
        state.daily_pnl = -20.0   # 4% loss on $500
        ok, reason = check_all("BTCUSDT", "LONG")
        assert ok is False
        assert "Daily loss" in reason

    def test_correlation_block(self):
        from app.state import OpenTrade
        state.open_trades["BTCUSDT"] = OpenTrade(
            trade_id="1", symbol="BTCUSDT", side="LONG",
            entry_price=100, stop_loss=90, take_profit=120,
            lot_size=0.1, opened_at="", strategy_version="v1.0"
        )
        ok, reason = check_all("ETHUSDT", "LONG")
        assert ok is False
        assert "Correlated" in reason

    def test_clean_state_allows_trade(self):
        ok, reason = check_all("BTCUSDT", "LONG")
        assert ok is True
        assert reason == "OK"
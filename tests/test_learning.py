import pytest
from datetime import datetime, timezone
from app.ai.features import build_feature_vector, FEATURE_COLUMNS, vector_to_list
from app.ai.validator import evaluate, should_deploy


def _make_indicators(price=50000.0):
    return {
        "price": price, "ema50": 49500.0, "ema200": 47000.0,
        "rsi": 52.0, "atr": 500.0, "volume_ratio": 1.3,
        "candle_body_pct": 0.7, "is_bullish_candle": True,
        "price_vs_ema50": 1.01, "trend_strength": 5.3,
        "volatility_pct": 1.0, "ema_gap_pct": 5.3, "ema50_slope": 0.1,
    }


class TestFeatureEngineering:
    def test_feature_vector_has_all_columns(self):
        ind_1h = _make_indicators()
        ind_4h = _make_indicators(price=50000)
        ts     = datetime.now(timezone.utc)
        feats  = build_feature_vector(ind_1h, ind_4h, ts)
        for col in FEATURE_COLUMNS:
            assert col in feats, f"Missing feature: {col}"

    def test_vector_to_list_length(self):
        ind_1h = _make_indicators()
        ind_4h = _make_indicators()
        ts     = datetime.now(timezone.utc)
        feats  = build_feature_vector(ind_1h, ind_4h, ts)
        vec    = vector_to_list(feats)
        assert len(vec) == len(FEATURE_COLUMNS)

    def test_all_values_are_numeric(self):
        ind_1h = _make_indicators()
        ind_4h = _make_indicators()
        ts     = datetime.now(timezone.utc)
        feats  = build_feature_vector(ind_1h, ind_4h, ts)
        vec    = vector_to_list(feats)
        for v in vec:
            assert isinstance(v, (int, float)), f"Non-numeric value: {v}"

    def test_trend_4h_encoding(self):
        ind_1h = _make_indicators()
        ind_4h_bull = {"ema50": 102.0, "ema200": 98.0, **_make_indicators()}
        ind_4h_bear = {"ema50": 95.0,  "ema200": 100.0, **_make_indicators()}

        ts   = datetime.now(timezone.utc)
        bull = build_feature_vector(ind_1h, ind_4h_bull, ts)
        bear = build_feature_vector(ind_1h, ind_4h_bear, ts)

        assert bull["trend_4h"] == 1.0
        assert bear["trend_4h"] == -1.0

    def test_time_context_encoding(self):
        ind_1h = _make_indicators()
        ind_4h = _make_indicators()
        ts     = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)  # Monday 14:00
        feats  = build_feature_vector(ind_1h, ind_4h, ts)
        assert feats["hour_utc"]    == 14.0
        assert feats["day_of_week"] == 0.0   # Monday


class TestModelValidator:
    def _make_dummy_model(self, accuracy: float):
        """Creates a mock model object with predict and predict_proba."""
        import numpy as np

        class MockModel:
            def __init__(self, acc):
                self._acc = acc
                self.feature_importances_ = np.ones(19) / 19

            def predict(self, X):
                n = len(X)
                preds = [1] * int(n * self._acc) + [0] * (n - int(n * self._acc))
                return np.array(preds)

            def predict_proba(self, X):
                n = len(X)
                return np.array([[0.3, 0.7]] * n)

        return MockModel(accuracy)

    def test_evaluate_returns_all_metrics(self):
        model  = self._make_dummy_model(0.6)
        X_val  = [[0.1] * 19] * 20
        y_val  = [1, 0] * 10
        result = evaluate(model, X_val, y_val)
        assert "accuracy"  in result
        assert "precision" in result
        assert "recall"    in result
        assert "f1_score"  in result

    def test_should_deploy_below_threshold(self):
        metrics = {"accuracy": 0.50, "f1_score": 0.48}
        deploy, reason = should_deploy(metrics, current_accuracy=None)
        assert deploy is False
        assert "below minimum" in reason

    def test_should_deploy_passes_threshold(self):
        metrics = {"accuracy": 0.62, "f1_score": 0.60, "precision": 0.61, "recall": 0.59}
        deploy, reason = should_deploy(metrics, current_accuracy=None)
        assert deploy is True

    def test_should_deploy_no_improvement(self):
        metrics = {"accuracy": 0.57, "f1_score": 0.55, "precision": 0.56, "recall": 0.54}
        deploy, reason = should_deploy(metrics, current_accuracy=0.56)
        assert deploy is False
        assert "Improvement" in reason

    def test_should_deploy_with_improvement(self):
        metrics = {"accuracy": 0.62, "f1_score": 0.60, "precision": 0.61, "recall": 0.59}
        deploy, reason = should_deploy(metrics, current_accuracy=0.56)
        assert deploy is True
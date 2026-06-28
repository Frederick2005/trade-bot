import os
import joblib
import numpy as np
from loguru import logger
from app.config import AI
from app.ai.features import FEATURE_COLUMNS, vector_to_list

_model = None
_model_version: str | None = None


def load_model(path: str, version: str) -> bool:
    global _model, _model_version
    try:
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        _model = joblib.load(path)
        _model_version = version
        logger.info(f"AI model loaded: {version} from {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def is_loaded() -> bool:
    return _model is not None


def get_version() -> str | None:
    return _model_version


def predict(features: dict) -> tuple[float, str]:
    """
    Returns (confidence, signal) where:
        confidence = probability the trade is a winner (0.0 – 1.0)
        signal     = 'BUY', 'SELL', or 'FLAT'
    Returns (0.0, 'FLAT') if model not loaded.
    """
    if _model is None:
        logger.debug("Model not loaded — returning neutral confidence")
        return 0.0, "FLAT"

    try:
        vec = vector_to_list(features)
        arr = np.array(vec).reshape(1, -1)
        proba = _model.predict_proba(arr)[0]

        # proba[1] = probability of class 1 (win)
        confidence = float(proba[1])

        if confidence >= AI.min_confidence:
            signal = "BUY"
        elif confidence <= (1 - AI.min_confidence):
            signal = "SELL"
        else:
            signal = "FLAT"

        logger.debug(f"AI prediction: confidence={confidence:.2%} signal={signal}")
        return confidence, signal

    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return 0.0, "FLAT"


def get_feature_importance() -> dict:
    """Returns feature importance scores from the trained model."""
    if _model is None:
        return {}
    try:
        scores = _model.feature_importances_
        return dict(zip(FEATURE_COLUMNS, [float(s) for s in scores]))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return {}


def unload() -> None:
    global _model, _model_version
    _model = None
    _model_version = None
    logger.info("AI model unloaded")
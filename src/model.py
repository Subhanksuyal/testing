import os
import joblib
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

try:
    from .features import add_derived_features, get_extended_features
except (ImportError, ValueError):
    from features import add_derived_features, get_extended_features

logger = logging.getLogger("ThermalRiskIntelligence")


class RiskPredictor:
    """
    ML Predictor for Industrial Thermal Anomaly Risk.
    Loads trained RandomForest Pipeline (with integrated Imputer + Scaler),
    calculates risk probabilities, and categorizes into LOW / MEDIUM / HIGH risk bands.
    """
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root_dir, 'models', 'industrial_risk_model.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please run 'python src/train_model.py' first.")

        model_payload = joblib.load(model_path)
        self.pipeline = model_payload['pipeline']
        self.features = model_payload['features']
        self.default_threshold = model_payload.get('threshold', 0.30)
        self.metadata = self._load_metadata(os.path.dirname(model_path))

    def _load_metadata(self, models_dir: str) -> Dict[str, Any]:
        meta_path = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}

    def predict(self, input_data: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Predicts industrial risk for a single thermal anomaly observation.
        """
        threshold = custom_threshold if custom_threshold is not None else self.default_threshold
        df_single = pd.DataFrame([input_data])
        df_feat = add_derived_features(df_single)

        # Align with model training feature set
        for col in self.features:
            if col not in df_feat.columns:
                df_feat[col] = np.nan

        X = df_feat[self.features]
        prob = float(self.pipeline.predict_proba(X)[0, 1])

        if prob < 0.30:
            risk_band = "LOW"
        elif 0.30 <= prob < 0.60:
            risk_band = "MEDIUM"
        else:
            risk_band = "HIGH"

        investigation_status = "HIGH / INVESTIGATE" if prob >= threshold else "LOW / MONITOR"

        return {
            'risk_probability': round(prob, 4),
            'risk_probability_pct': round(prob * 100, 1),
            'risk_band': risk_band,
            'investigation_status': investigation_status,
            'prediction': 1 if prob >= threshold else 0,
            'used_threshold': threshold
        }

    def predict_batch(self, df: pd.DataFrame, custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Predicts industrial risk probabilities and bands for a full DataFrame of observations.
        """
        if df.empty:
            df_empty = df.copy()
            df_empty['risk_probability'] = pd.Series(dtype=float)
            df_empty['risk_probability_pct'] = pd.Series(dtype=float)
            df_empty['risk_band'] = pd.Series(dtype=str)
            df_empty['investigation_status'] = pd.Series(dtype=str)
            return df_empty

        threshold = custom_threshold if custom_threshold is not None else self.default_threshold
        df_out = add_derived_features(df)

        for col in self.features:
            if col not in df_out.columns:
                df_out[col] = np.nan

        X = df_out[self.features]
        probs = self.pipeline.predict_proba(X)[:, 1]

        df_out['risk_probability'] = np.round(probs, 4)
        df_out['risk_probability_pct'] = np.round(probs * 100.0, 1)

        # Vectorized Risk Band classification
        conditions = [
            df_out['risk_probability'] < 0.30,
            (df_out['risk_probability'] >= 0.30) & (df_out['risk_probability'] < 0.60),
            df_out['risk_probability'] >= 0.60
        ]
        choices = ['LOW', 'MEDIUM', 'HIGH']
        df_out['risk_band'] = np.select(conditions, choices, default='LOW')
        df_out['investigation_status'] = np.where(df_out['risk_probability'] >= threshold, 'HIGH / INVESTIGATE', 'LOW / MONITOR')

        return df_out

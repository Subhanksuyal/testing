import os
import joblib
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

try:
    from .features import add_derived_features, get_base_features
except (ImportError, ValueError):
    from features import add_derived_features, get_base_features

logger = logging.getLogger("ThermalRiskIntelligence")

REQUIRED_FEATURES: List[str] = [
    'frp',
    'confidence',
    'industry_distance_km',
    'forest_distance_km',
    'persistence_days',
    'night_flag'
]


def _patch_imputer_compatibility(pipeline: Any) -> None:
    """
    Patches unpickled SimpleImputer instances to avoid AttributeError
    on private attributes (e.g. _fill_dtype, _fit_dtype) across differing scikit-learn versions.
    """
    if hasattr(pipeline, 'named_steps'):
        if 'imputer' in pipeline.named_steps:
            imputer = pipeline.named_steps['imputer']
            if not hasattr(imputer, '_fill_dtype'):
                setattr(imputer, '_fill_dtype', np.float64)
            if not hasattr(imputer, '_fit_dtype'):
                setattr(imputer, '_fit_dtype', np.float64)
            if not hasattr(imputer, 'keep_empty_features'):
                setattr(imputer, 'keep_empty_features', False)


class RiskPredictor:
    """
    ML Predictor for Industrial Thermal Anomaly Risk.
    Loads trained RandomForest Pipeline (with integrated Imputer + Scaler),
    calculates risk probabilities, and categorizes into LOW / MEDIUM / HIGH risk bands.
    """
    def __init__(self, model_path: Optional[str] = None):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(root_dir, 'models', 'industrial_risk_model.joblib')

        self.root_dir = root_dir
        self.model_path = model_path
        self.features = list(REQUIRED_FEATURES)
        self.default_threshold = 0.30
        self.metadata = {}
        self.pipeline = None

        if os.path.exists(model_path):
            try:
                model_payload = joblib.load(model_path)
                self.pipeline = model_payload.get('pipeline')
                if 'features' in model_payload:
                    self.features = model_payload['features']
                self.default_threshold = model_payload.get('threshold', 0.30)
                _patch_imputer_compatibility(self.pipeline)
            except Exception as e:
                logger.warning(f"Failed to load pickled model ({e}). Retraining from practice dataset...")
                self._train_fallback_pipeline()
        else:
            logger.info("Model file not found. Training initial model from practice dataset...")
            self._train_fallback_pipeline()

        self.metadata = self._load_metadata(os.path.dirname(model_path))

    def _train_fallback_pipeline(self) -> None:
        """Trains a fresh pipeline from data/practice_firms.csv if pickle loading fails."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier

            data_path = os.path.join(self.root_dir, 'data', 'practice_firms.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Practice data not found at {data_path}")

            df = pd.read_csv(data_path)
            X = pd.DataFrame()
            for col in self.features:
                if col in df.columns:
                    X[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                else:
                    X[col] = np.nan
            y = pd.to_numeric(df['industrial_risk_label'], errors='coerce').astype(int)

            self.pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1))
            ])
            self.pipeline.fit(X, y)
            _patch_imputer_compatibility(self.pipeline)
            logger.info("Fallback model successfully trained in memory.")
        except Exception as ex:
            logger.error(f"Fallback model training failed: {ex}")
            raise ex

    def _load_metadata(self, models_dir: str) -> Dict[str, Any]:
        meta_path = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}

    def _prepare_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts, converts to numeric float, and orders all required feature columns.
        Explicitly excludes coordinates (latitude, longitude) from model features.
        """
        # Ensure derived/missing fields like persistence_days and night_flag are populated
        df_prepared = add_derived_features(df)

        X = pd.DataFrame(index=df_prepared.index)
        for col in self.features:
            if col in df_prepared.columns:
                X[col] = pd.to_numeric(df_prepared[col], errors='coerce').astype(float)
            else:
                X[col] = np.nan

        return X[self.features]

    def _safe_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Safely computes class probabilities with fallback handling for sklearn version mismatches.
        """
        _patch_imputer_compatibility(self.pipeline)
        try:
            return self.pipeline.predict_proba(X)[:, 1]
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Direct predict_proba failed ({e}). Running manual step-by-step imputation fallback...")
            try:
                # Step-by-step pipeline execution
                X_vals = X.values.astype(float)
                if hasattr(self.pipeline, 'named_steps'):
                    imputer = self.pipeline.named_steps.get('imputer')
                    scaler = self.pipeline.named_steps.get('scaler')
                    clf = self.pipeline.named_steps.get('clf')

                    # Impute using statistics_ if available
                    if imputer and hasattr(imputer, 'statistics_'):
                        stats = imputer.statistics_
                        for i in range(X_vals.shape[1]):
                            mask = np.isnan(X_vals[:, i])
                            X_vals[mask, i] = stats[i]
                    else:
                        X_vals = np.nan_to_num(X_vals, nan=0.0)

                    if scaler:
                        X_vals = scaler.transform(X_vals)

                    if clf:
                        return clf.predict_proba(X_vals)[:, 1]

                # If steps not accessible, retrain fresh pipeline
                self._train_fallback_pipeline()
                return self.pipeline.predict_proba(X)[:, 1]
            except Exception as final_ex:
                logger.error(f"Fallback predict_proba also failed: {final_ex}")
                raise final_ex

    def predict(self, input_data: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Predicts industrial risk for a single thermal anomaly observation.
        """
        threshold = custom_threshold if custom_threshold is not None else self.default_threshold
        df_single = pd.DataFrame([input_data])
        X = self._prepare_feature_matrix(df_single)

        probs = self._safe_predict_proba(X)
        prob = float(probs[0])

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
        X = self._prepare_feature_matrix(df_out)

        probs = self._safe_predict_proba(X)

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

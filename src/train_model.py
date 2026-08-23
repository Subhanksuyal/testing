import os
import json
import joblib
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

try:
    from .features import add_derived_features, get_base_features
    from .utils import validate_fire_dataframe
except (ImportError, ValueError):
    from features import add_derived_features, get_base_features
    from utils import validate_fire_dataframe

DEFAULT_RISK_THRESHOLD = 0.30

# Canonical 6 features in exact required order
FEATURE_COLUMNS = [
    'frp',
    'confidence',
    'industry_distance_km',
    'forest_distance_km',
    'persistence_days',
    'night_flag'
]


def train_and_save_model():
    """
    Loads practice data, engineers features, trains a robust Random Forest model
    pipeline with missing value imputation, evaluates performance, and saves model & metadata.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'practice_firms.csv')
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    raw_df = pd.read_csv(data_path)
    df, diag = validate_fire_dataframe(raw_df)
    print(f"Loaded and validated dataset: {df.shape} (Valid: {diag['valid_count']}, Dropped: {diag['removed_count']})")

    # Feature Engineering
    df_feat = add_derived_features(df)
    target_col = 'industrial_risk_label'

    if target_col not in df_feat.columns:
        raise KeyError(f"Target column '{target_col}' not found in training dataset.")

    # Convert all feature columns strictly to float numeric
    X = pd.DataFrame()
    for col in FEATURE_COLUMNS:
        if col in df_feat.columns:
            X[col] = pd.to_numeric(df_feat[col], errors='coerce').astype(float)
        else:
            X[col] = np.nan

    y = pd.to_numeric(df_feat[target_col], errors='coerce').astype(int)

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Positive class ratio: Train={y_train.mean():.3f}, Test={y_test.mean():.3f}")

    # Build Pipeline with Imputer for missing GIS / FRP values, Scaler, and RandomForest
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('clf', clf)
    ])

    print("Training Random Forest Classifier...")
    pipeline.fit(X_train, y_train)

    # Predictions & Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DEFAULT_RISK_THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print(f"MODEL EVALUATION (Threshold = {DEFAULT_RISK_THRESHOLD})")
    print("=" * 50)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC AUC:         {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 50)

    # Feature Importances
    rf_model = pipeline.named_steps['clf']
    importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)}
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    # Save Model Artifact
    model_path = os.path.join(models_dir, 'industrial_risk_model.joblib')
    model_payload = {
        'pipeline': pipeline,
        'features': FEATURE_COLUMNS,
        'base_features': FEATURE_COLUMNS,
        'threshold': DEFAULT_RISK_THRESHOLD,
        'training_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(model_payload, model_path)
    print(f"Model successfully saved to: {model_path}")

    # Save Model Metadata & Evaluation Report
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    metadata = {
        "model_name": "RandomForestClassifier (Industrial Thermal Anomaly Risk)",
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features": FEATURE_COLUMNS,
        "features_count": len(FEATURE_COLUMNS),
        "default_threshold": DEFAULT_RISK_THRESHOLD,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "feature_importances": sorted_importances,
        "class_distribution": {
            "negative_0": int((y == 0).sum()),
            "positive_1": int((y == 1).sum())
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to: {metadata_path}")
    return metadata


if __name__ == "__main__":
    train_and_save_model()

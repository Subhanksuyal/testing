import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from features import (
    add_derived_features,
    get_base_features,
    get_extended_features,
    compute_proximity_score,
    extract_night_flag,
    calculate_persistence_days
)
from model import RiskPredictor


def test_compute_proximity_score():
    s = pd.Series([0.0, 5.0, 10.0, np.nan])
    scores = compute_proximity_score(s, scale=5.0)
    assert pytest.approx(scores.iloc[0], 0.01) == 1.0
    assert pytest.approx(scores.iloc[1], 0.01) == np.exp(-1.0) # ~0.3678
    assert scores.iloc[3] == 0.0 # NaN mapped to 0.0


def test_extract_night_flag():
    # Test daynight tag
    assert extract_night_flag(pd.Series({'daynight': 'N'})) == 1
    assert extract_night_flag(pd.Series({'daynight': 'D'})) == 0
    assert extract_night_flag(pd.Series({'daynight': 'n'})) == 1
    
    # Test fallback from acq_time (18:30 UTC -> 00:00 IST -> Night)
    assert extract_night_flag(pd.Series({'acq_time': '1830'})) == 1
    # 06:00 UTC -> 11:30 IST -> Day
    assert extract_night_flag(pd.Series({'acq_time': '0600'})) == 0


def test_persistence_calculation():
    df = pd.DataFrame({
        'latitude': [28.61, 28.61, 19.07],
        'longitude': [77.20, 77.20, 72.87],
        'acq_date': ['2026-08-20', '2026-08-21', '2026-08-20']
    })
    pers = calculate_persistence_days(df)
    # First cluster has 2 distinct dates
    assert pers.iloc[0] == 2
    assert pers.iloc[1] == 2
    # Second cluster has 1 date
    assert pers.iloc[2] == 1


def test_model_prediction_single_and_batch():
    predictor = RiskPredictor()

    # High risk profile
    high_risk_input = {
        'latitude': 22.0,
        'longitude': 78.0,
        'frp': 450.0,
        'confidence': 95.0,
        'industry_distance_km': 0.8,
        'forest_distance_km': 15.0,
        'persistence_days': 5,
        'night_flag': 1
    }
    result = predictor.predict(high_risk_input)
    assert 'risk_probability' in result
    assert 'risk_band' in result
    assert result['risk_probability'] >= 0.0
    assert result['risk_probability'] <= 1.0
    assert result['risk_band'] in ['LOW', 'MEDIUM', 'HIGH']

    # Batch test with missing GIS distances (NaN values)
    df_batch = pd.DataFrame([
        {'latitude': 28.6, 'longitude': 77.2, 'frp': 300.0, 'confidence': 90.0, 'industry_distance_km': np.nan, 'forest_distance_km': np.nan},
        {'latitude': 19.1, 'longitude': 72.8, 'frp': 50.0, 'confidence': 50.0, 'industry_distance_km': 12.0, 'forest_distance_km': 1.0}
    ])
    df_pred = predictor.predict_batch(df_batch)
    assert len(df_pred) == 2
    assert 'risk_probability' in df_pred.columns
    assert 'risk_band' in df_pred.columns
    assert df_pred['risk_probability'].notna().all()

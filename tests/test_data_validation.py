import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils import validate_fire_dataframe, normalize_column_names, filter_by_region, calculate_distance_km


def test_normalize_column_names():
    df_raw = pd.DataFrame({
        'LATITUDE': [22.5],
        'LONGITUDE': [78.2],
        'FRP': [150.0],
        'CONFIDENCE': [85.0],
        'ACQ_DATE': ['2026-08-23'],
        'ACQ_TIME': ['1430'],
        'DAYNIGHT': ['N']
    })
    df_norm = normalize_column_names(df_raw)
    assert 'latitude' in df_norm.columns
    assert 'longitude' in df_norm.columns
    assert 'frp' in df_norm.columns
    assert 'confidence' in df_norm.columns
    assert 'acq_date' in df_norm.columns
    assert 'daynight' in df_norm.columns


def test_validate_fire_dataframe_clean():
    df_sample = pd.DataFrame({
        'latitude': [28.6139, 19.0760],
        'longitude': [77.2090, 72.8777],
        'frp': [45.2, 120.0],
        'confidence': [80.0, 95.0]
    })
    df_clean, diag = validate_fire_dataframe(df_sample)
    assert len(df_clean) == 2
    assert diag['total_fetched'] == 2
    assert diag['valid_count'] == 2
    assert diag['removed_count'] == 0


def test_validate_fire_dataframe_invalid_coordinates():
    df_invalid = pd.DataFrame({
        'latitude': [28.6139, np.nan, 120.0, 20.0],  # 120.0 is > 90 lat
        'longitude': [77.2090, 72.8777, 75.0, 250.0], # 250.0 is > 180 lon
        'frp': [10.0, 20.0, 30.0, 40.0]
    })
    df_clean, diag = validate_fire_dataframe(df_invalid)
    # Only row 0 is valid
    assert len(df_clean) == 1
    assert diag['valid_count'] == 1
    assert diag['removed_count'] == 3


def test_validate_fire_dataframe_missing_required_cols():
    df_no_coords = pd.DataFrame({
        'temperature': [300.0],
        'frp': [50.0]
    })
    df_clean, diag = validate_fire_dataframe(df_no_coords)
    assert df_clean.empty
    assert diag['valid_count'] == 0
    assert len(diag['reasons']) > 0


def test_filter_by_region():
    df = pd.DataFrame({
        'latitude': [28.61, 19.07, 30.31], # Delhi, Mumbai, Uttarakhand (Dehradun)
        'longitude': [77.20, 72.87, 78.03]
    })
    delhi_df = filter_by_region(df, "Delhi (NCR)")
    assert len(delhi_df) == 1
    assert delhi_df.iloc[0]['latitude'] == 28.61

    all_india_df = filter_by_region(df, "India (All)")
    assert len(all_india_df) == 3


def test_calculate_distance_km():
    # Distance between Delhi (28.6139, 77.2090) and Mumbai (19.0760, 72.8777) is ~1148 km
    dist = calculate_distance_km(28.6139, 77.2090, 19.0760, 72.8777)
    assert dist is not None
    assert 1140 < dist < 1160

    # Missing coordinate test
    assert calculate_distance_km(None, 77.0, 19.0, 72.0) is None
    assert calculate_distance_km(28.0, np.nan, 19.0, 72.0) is None

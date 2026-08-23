import pandas as pd
import numpy as np
from typing import List

BASE_FEATURES: List[str] = [
    'frp',
    'confidence',
    'industry_distance_km',
    'forest_distance_km',
    'persistence_days',
    'night_flag'
]

EXTENDED_FEATURES: List[str] = BASE_FEATURES + [
    'industry_proximity_score',
    'forest_proximity_score',
    'thermal_intensity_score',
    'confidence_score',
    'persistence_score'
]


def get_base_features() -> List[str]:
    return list(BASE_FEATURES)


def get_extended_features() -> List[str]:
    return list(EXTENDED_FEATURES)


def compute_proximity_score(distance_km: pd.Series, scale: float = 5.0) -> pd.Series:
    """
    Computes an exponential proximity score where smaller distances yield higher scores.
    Handles NaN by returning 0.0 (meaning infinitely far or unknown).
    """
    # Score = exp(-distance / scale), 0 distance -> 1.0, 5km -> ~0.37, 10km -> ~0.13, 20km -> ~0.018
    score = np.exp(-distance_km / scale)
    return score.fillna(0.0)


def extract_night_flag(row: pd.Series) -> int:
    """
    Determines day (0) vs night (1) observation based on FIRMS daynight field
    or UTC acquisition time approximation for India (UTC+5:30).
    """
    # Check explicit daynight field
    if 'daynight' in row and pd.notna(row['daynight']):
        dn = str(row['daynight']).strip().upper()
        if dn == 'N' or dn == 'NIGHT' or dn == '1':
            return 1
        elif dn == 'D' or dn == 'DAY' or dn == '0':
            return 0

    # If night_flag already exists as a numeric column
    if 'night_flag' in row and pd.notna(row['night_flag']):
        try:
            return 1 if int(row['night_flag']) == 1 else 0
        except (ValueError, TypeError):
            pass

    # Approximation from acq_time (UTC format 'HHMM' e.g. '1830')
    if 'acq_time' in row and pd.notna(row['acq_time']):
        try:
            t_str = str(row['acq_time']).strip().zfill(4)
            utc_hour = int(t_str[:2])
            utc_min = int(t_str[2:])
            # Convert to Indian Standard Time (IST = UTC + 5:30)
            ist_hour = (utc_hour + 5 + (1 if utc_min + 30 >= 60 else 0)) % 24
            # Night is approximately between 19:00 (7 PM) and 05:00 (5 AM)
            if ist_hour >= 19 or ist_hour < 5:
                return 1
            else:
                return 0
        except Exception:
            pass

    # Default to day if no time information
    return 0


def calculate_persistence_days(df: pd.DataFrame, spatial_tol_deg: float = 0.015) -> pd.Series:
    """
    Calculates the persistence of thermal anomalies across distinct acquisition dates.
    Spatially clusters nearby observations (~1.5 km tolerance) and counts distinct dates.
    If multi-day observations are unavailable or persistence_days already exists, uses that.
    """
    if df.empty:
        return pd.Series(dtype=float)

    if 'persistence_days' in df.columns and df['persistence_days'].notna().all():
        return df['persistence_days'].astype(int)

    if 'acq_date' not in df.columns or df['acq_date'].nunique() <= 1:
        # Single day snapshot: fallback persistence is 1
        if 'persistence_days' in df.columns:
            return df['persistence_days'].fillna(1).astype(int)
        return pd.Series(1, index=df.index, dtype=int)

    # Spatially bucket coordinates
    lat_grid = (df['latitude'] / spatial_tol_deg).round()
    lon_grid = (df['longitude'] / spatial_tol_deg).round()
    spatial_id = lat_grid.astype(str) + "_" + lon_grid.astype(str)

    # Group by spatial cluster and count unique dates
    date_counts = df.groupby(spatial_id)['acq_date'].nunique().to_dict()
    persistence = spatial_id.map(date_counts).fillna(1).astype(int)
    return persistence


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features to the dataframe for ML modeling.
    Preserves original columns and handles missing values safely.
    """
    df_out = df.copy()

    # Ensure base columns exist or create defaults
    if 'night_flag' not in df_out.columns or df_out['night_flag'].isna().any():
        df_out['night_flag'] = df_out.apply(extract_night_flag, axis=1)

    if 'persistence_days' not in df_out.columns or df_out['persistence_days'].isna().any():
        df_out['persistence_days'] = calculate_persistence_days(df_out)

    if 'industry_distance_km' not in df_out.columns:
        df_out['industry_distance_km'] = np.nan

    if 'forest_distance_km' not in df_out.columns:
        df_out['forest_distance_km'] = np.nan

    if 'frp' not in df_out.columns:
        df_out['frp'] = np.nan

    if 'confidence' not in df_out.columns:
        df_out['confidence'] = np.nan

    # Derived Features:
    # 1. Proximity scores using exponential decay
    df_out['industry_proximity_score'] = compute_proximity_score(df_out['industry_distance_km'], scale=5.0)
    df_out['forest_proximity_score'] = compute_proximity_score(df_out['forest_distance_km'], scale=5.0)

    # 2. Normalized FRP (assuming 500 MW typical cap)
    df_out['thermal_intensity_score'] = (df_out['frp'] / 500.0).clip(lower=0.0, upper=1.0).fillna(0.0)

    # 3. Normalized Confidence
    df_out['confidence_score'] = (df_out['confidence'] / 100.0).clip(lower=0.0, upper=1.0).fillna(0.5)

    # 4. Normalized Persistence
    df_out['persistence_score'] = (df_out['persistence_days'] / 10.0).clip(lower=0.0, upper=1.0).fillna(0.1)

    return df_out

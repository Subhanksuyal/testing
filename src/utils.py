import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from geopy.distance import geodesic

logger = logging.getLogger("ThermalRiskIntelligence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Bounding boxes for Indian States & Regions [min_lat, max_lat, min_lon, max_lon]
INDIAN_REGIONS: Dict[str, Dict[str, float]] = {
    "India (All)": {"min_lat": 6.5, "max_lat": 37.5, "min_lon": 68.0, "max_lon": 97.5},
    "Uttarakhand": {"min_lat": 28.7, "max_lat": 31.5, "min_lon": 77.5, "max_lon": 81.1},
    "Delhi (NCR)": {"min_lat": 28.4, "max_lat": 28.9, "min_lon": 76.8, "max_lon": 77.4},
    "Uttar Pradesh": {"min_lat": 23.8, "max_lat": 30.4, "min_lon": 77.0, "max_lon": 84.7},
    "Haryana": {"min_lat": 27.6, "max_lat": 30.9, "min_lon": 74.4, "max_lon": 77.6},
    "Rajasthan": {"min_lat": 23.0, "max_lat": 30.2, "min_lon": 69.5, "max_lon": 78.3},
    "Maharashtra": {"min_lat": 15.6, "max_lat": 22.0, "min_lon": 72.6, "max_lon": 80.9},
    "Gujarat": {"min_lat": 20.1, "max_lat": 24.7, "min_lon": 68.1, "max_lon": 74.5},
    "West Bengal": {"min_lat": 21.5, "max_lat": 27.2, "min_lon": 85.8, "max_lon": 89.9},
    "Jharkhand": {"min_lat": 21.9, "max_lat": 25.3, "min_lon": 83.3, "max_lon": 87.9},
    "Odisha": {"min_lat": 17.8, "max_lat": 22.6, "min_lon": 81.4, "max_lon": 87.5},
    "Chhattisgarh": {"min_lat": 17.8, "max_lat": 24.1, "min_lon": 80.2, "max_lon": 84.4},
    "Madhya Pradesh": {"min_lat": 21.1, "max_lat": 26.9, "min_lon": 74.0, "max_lon": 82.8},
    "Punjab": {"min_lat": 29.5, "max_lat": 32.5, "min_lon": 73.8, "max_lon": 76.9},
    "Bihar": {"min_lat": 24.3, "max_lat": 27.5, "min_lon": 83.3, "max_lon": 88.3},
    "Tamil Nadu": {"min_lat": 8.0, "max_lat": 13.6, "min_lon": 76.2, "max_lon": 80.4},
    "Karnataka": {"min_lat": 11.5, "max_lat": 18.5, "min_lon": 74.0, "max_lon": 78.6},
    "Andhra Pradesh": {"min_lat": 12.6, "max_lat": 19.9, "min_lon": 76.7, "max_lon": 84.8},
    "Telangana": {"min_lat": 15.8, "max_lat": 19.9, "min_lon": 77.2, "max_lon": 81.8},
    "Assam": {"min_lat": 24.1, "max_lat": 28.0, "min_lon": 89.7, "max_lon": 96.0},
    "Himachal Pradesh": {"min_lat": 30.3, "max_lat": 33.2, "min_lon": 75.5, "max_lon": 79.0},
    "Jammu & Kashmir": {"min_lat": 32.2, "max_lat": 37.1, "min_lon": 73.8, "max_lon": 80.3}
}

# Column normalization mappings across multiple FIRMS / MODIS / VIIRS formats
COLUMN_ALIASES = {
    'latitude': ['latitude', 'lat', 'lat_deg', 'LATITUDE', 'LAT'],
    'longitude': ['longitude', 'lon', 'long', 'lon_deg', 'LONGITUDE', 'LON'],
    'frp': ['frp', 'FRP', 'fire_radiative_power', 'power', 'frp_mw'],
    'confidence': ['confidence', 'CONFIDENCE', 'confidence_pct', 'conf', 'scan_confidence', 'confidence_cat'],
    'acq_date': ['acq_date', 'ACQ_DATE', 'date', 'acquisition_date'],
    'acq_time': ['acq_time', 'ACQ_TIME', 'time', 'acquisition_time'],
    'satellite': ['satellite', 'SATELLITE', 'sat', 'satellite_name'],
    'instrument': ['instrument', 'INSTRUMENT', 'sensor'],
    'daynight': ['daynight', 'DAYNIGHT', 'day_night', 'dn', 'DN'],
    'bright_ti4': ['bright_ti4', 'brightness', 'bright_t31', 'BRIGHT_TI4'],
    'scan': ['scan', 'SCAN'],
    'track': ['track', 'TRACK'],
    'version': ['version', 'VERSION']
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names using case-insensitive mapping.
    """
    if df.empty:
        return df

    df_renamed = df.copy()
    current_cols = {str(c).strip().lower(): c for c in df_renamed.columns}
    
    rename_dict = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in current_cols:
                original_col = current_cols[alias_lower]
                rename_dict[original_col] = canonical_name
                break

    df_renamed = df_renamed.rename(columns=rename_dict)
    return df_renamed


def validate_fire_dataframe(df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates fire anomaly observations DataFrame:
    - Normalizes column names
    - Checks required coordinates
    - Validates latitude (-90 to 90), longitude (-180 to 180)
    - Validates FRP (>= 0)
    - Validates confidence (0 to 100)
    - Returns clean DataFrame and detailed diagnostics metrics.
    """
    diagnostics = {
        'total_fetched': 0,
        'valid_count': 0,
        'removed_count': 0,
        'reasons': []
    }

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        diagnostics['reasons'].append("Empty or null dataframe provided.")
        return pd.DataFrame(), diagnostics

    diagnostics['total_fetched'] = len(df)
    df_clean = normalize_column_names(df)

    # 1. Coordinate column presence check
    if 'latitude' not in df_clean.columns or 'longitude' not in df_clean.columns:
        diagnostics['reasons'].append(f"Missing required coordinate columns 'latitude'/'longitude'. Found: {list(df.columns)}")
        diagnostics['removed_count'] = len(df)
        return pd.DataFrame(), diagnostics

    # Convert coordinates to numeric, turning invalid strings into NaN
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')

    # Filter invalid / NaN coordinates
    valid_coords_mask = (
        df_clean['latitude'].notna() &
        df_clean['longitude'].notna() &
        (df_clean['latitude'] >= -90.0) &
        (df_clean['latitude'] <= 90.0) &
        (df_clean['longitude'] >= -180.0) &
        (df_clean['longitude'] <= 180.0)
    )

    invalid_coords_count = (~valid_coords_mask).sum()
    if invalid_coords_count > 0:
        diagnostics['reasons'].append(f"Dropped {invalid_coords_count} rows with invalid/missing coordinates.")

    df_clean = df_clean[valid_coords_mask].copy()

    # 2. Validate FRP
    if 'frp' in df_clean.columns:
        df_clean['frp'] = pd.to_numeric(df_clean['frp'], errors='coerce')
        negative_frp_mask = df_clean['frp'] < 0
        if negative_frp_mask.sum() > 0:
            df_clean.loc[negative_frp_mask, 'frp'] = np.nan
            diagnostics['reasons'].append(f"Fixed {negative_frp_mask.sum()} negative FRP values to NaN.")
    else:
        df_clean['frp'] = np.nan

    # 3. Validate confidence
    if 'confidence' in df_clean.columns:
        # VIIRS FIRMS uses categorical confidence ('l', 'n', 'h') or numeric 0-100
        def clean_confidence(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                v_lower = val.strip().lower()
                if v_lower == 'l' or v_lower == 'low':
                    return 30.0
                elif v_lower == 'n' or v_lower == 'nominal':
                    return 65.0
                elif v_lower == 'h' or v_lower == 'high':
                    return 90.0
                try:
                    num_val = float(val)
                    return num_val if 0 <= num_val <= 100 else np.nan
                except ValueError:
                    return np.nan
            try:
                num_val = float(val)
                return num_val if 0 <= num_val <= 100 else np.nan
            except (ValueError, TypeError):
                return np.nan

        df_clean['confidence'] = df_clean['confidence'].apply(clean_confidence)
    else:
        df_clean['confidence'] = np.nan

    # 4. Normalize Datetime
    if 'acq_date' in df_clean.columns:
        if 'acq_time' in df_clean.columns:
            def build_datetime(row):
                d_str = str(row['acq_date']).strip()
                t_str = str(row['acq_time']).strip().zfill(4)
                if len(t_str) == 4 and t_str.isdigit():
                    return f"{d_str} {t_str[:2]}:{t_str[2:]}:00"
                return d_str
            df_clean['acq_datetime'] = df_clean.apply(build_datetime, axis=1)
        else:
            df_clean['acq_datetime'] = df_clean['acq_date'].astype(str)
    elif 'acq_datetime' not in df_clean.columns:
        df_clean['acq_datetime'] = "N/A"

    diagnostics['valid_count'] = len(df_clean)
    diagnostics['removed_count'] = diagnostics['total_fetched'] - diagnostics['valid_count']

    logger.info(f"Data Validation: Fetched={diagnostics['total_fetched']}, Valid={diagnostics['valid_count']}, Removed={diagnostics['removed_count']}")
    return df_clean, diagnostics


def filter_by_region(df: pd.DataFrame, region_name: str, custom_bbox: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Filter observations within selected state/region or custom bounding box.
    """
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df

    bbox = None
    if custom_bbox is not None:
        bbox = custom_bbox
    elif region_name in INDIAN_REGIONS:
        bbox = INDIAN_REGIONS[region_name]

    if bbox is None:
        return df

    mask = (
        (df['latitude'] >= bbox['min_lat']) &
        (df['latitude'] <= bbox['max_lat']) &
        (df['longitude'] >= bbox['min_lon']) &
        (df['longitude'] <= bbox['max_lon'])
    )
    return df[mask].copy()


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Calculate geodesic distance between two points in km.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    try:
        if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
            return None
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except Exception:
        return None


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Fast vectorized Haversine distance calculation in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2 +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

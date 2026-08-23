import os
import io
import time
import logging
import datetime
import requests
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

try:
    from .utils import validate_fire_dataframe, normalize_column_names
except (ImportError, ValueError):
    from utils import validate_fire_dataframe, normalize_column_names

# Load .env if present
load_dotenv()

logger = logging.getLogger("ThermalRiskIntelligence")

# NASA FIRMS Supported Satellite Sources
FIRMS_SOURCES = {
    "VIIRS Suomi NPP (375m NRT)": "VIIRS_SNPP_NRT",
    "VIIRS NOAA-20 (375m NRT)": "VIIRS_NOAA20_NRT",
    "VIIRS NOAA-21 (375m NRT)": "VIIRS_NOAA21_NRT",
    "MODIS Terra/Aqua (1km NRT)": "MODIS_NRT"
}

# NASA Open NRT CSV Feeds (South Asia regional 24h/48h/7d active fire feeds)
OPEN_NRT_FEEDS = {
    "VIIRS_SNPP_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_NOAA20_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_NOAA21_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_South_Asia_24h.csv",
    "MODIS_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv"
}


def get_firms_api_key() -> Optional[str]:
    """
    Safely retrieves the NASA FIRMS MAP_KEY / API_KEY from:
    1. Streamlit Secrets (st.secrets)
    2. Environment variables (FIRMS_API_KEY, FIRMS_MAP_KEY)
    3. .env file
    Returns None if not configured.
    """
    # Try Streamlit Secrets first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "FIRMS_API_KEY" in st.secrets:
            key = st.secrets["FIRMS_API_KEY"]
            if key and str(key).strip() and not str(key).strip().startswith("YOUR_"):
                return str(key).strip()
    except Exception:
        pass

    # Try Environment variables
    for env_var in ["FIRMS_API_KEY", "FIRMS_MAP_KEY", "NASA_FIRMS_KEY"]:
        key = os.environ.get(env_var)
        if key and str(key).strip() and not str(key).strip().startswith("YOUR_"):
            return str(key).strip()

    return None


def _get_cache_path() -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, 'firms_last_valid_cache.csv')


def save_last_valid_cache(df: pd.DataFrame, source: str) -> None:
    """Saves the latest successful fetch to disk for offline / API failure fallback."""
    if df is not None and not df.empty:
        try:
            cache_file = _get_cache_path()
            df_to_save = df.copy()
            df_to_save['__cached_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_to_save['__cached_source'] = source
            df_to_save.to_csv(cache_file, index=False)
            logger.info(f"Saved {len(df)} rows to local FIRMS backup cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save FIRMS cache: {e}")


def load_last_valid_cache() -> Tuple[pd.DataFrame, Optional[str]]:
    """Loads the last successfully saved FIRMS cache."""
    cache_file = _get_cache_path()
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            cached_at = df['__cached_at'].iloc[0] if '__cached_at' in df.columns else "Unknown"
            # Clean internal columns
            drop_cols = [c for c in ['__cached_at', '__cached_source'] if c in df.columns]
            df = df.drop(columns=drop_cols)
            return df, cached_at
        except Exception as e:
            logger.warning(f"Failed to read FIRMS cache: {e}")
    return pd.DataFrame(), None


def fetch_firms_live_data(
    source_key: str = "VIIRS_SNPP_NRT",
    country_code: str = "IND",
    day_range: int = 1,
    bbox: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Fetches real-time active fire observations from NASA FIRMS.
    - Uses official Area / Country API if MAP_KEY is configured.
    - Seamlessly falls back to NASA FIRMS Open NRT South Asia feeds if key is missing/invalid.
    - Falls back to last disk cache if network/NASA API is down.
    - Validates and normalizes all rows.
    """
    api_key = get_firms_api_key()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    response_payload = {
        'success': False,
        'data': pd.DataFrame(),
        'source_used': source_key,
        'message': '',
        'last_fetched': now_str,
        'is_live': False,
        'is_cached': False,
        'diagnostics': {}
    }

    raw_df = None
    fetch_success = False
    source_description = source_key

    # 1. Try Official FIRMS API with MAP_KEY
    if api_key:
        try:
            day_range_clamped = max(1, min(10, int(day_range)))
            if bbox:
                w, s, e, n = bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat']
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source_key}/{w},{s},{e},{n}/{day_range_clamped}"
            else:
                url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/{source_key}/{country_code}/{day_range_clamped}"

            logger.info(f"Querying NASA FIRMS API (key configured)...")
            res = requests.get(url, timeout=20, headers={'User-Agent': 'IndustrialThermalRiskSystem/1.0 (SIH Project)'})

            if res.status_code == 200:
                text_data = res.text.strip()
                if text_data and not text_data.startswith("<!DOCTYPE") and not text_data.startswith("Error"):
                    raw_df = pd.read_csv(io.StringIO(text_data))
                    fetch_success = True
                    response_payload['message'] = f"Successfully fetched live data from NASA FIRMS API ({source_key}, {day_range} day)."
                    response_payload['is_live'] = True
                elif "Invalid MAP_KEY" in text_data or "Error" in text_data:
                    logger.warning(f"NASA FIRMS returned API error: {text_data}")
                    response_payload['message'] = f"NASA FIRMS API Key warning: {text_data}. Attempting open feed fallback."
            elif res.status_code == 401 or res.status_code == 403:
                logger.warning(f"NASA FIRMS 401/403 Authentication failed.")
                response_payload['message'] = "NASA FIRMS API key is invalid or expired. Attempting open feed fallback."
            elif res.status_code == 429:
                response_payload['message'] = "NASA FIRMS API rate limit reached (429). Attempting open feed fallback."
            else:
                response_payload['message'] = f"NASA FIRMS HTTP {res.status_code}. Attempting open feed fallback."

        except requests.exceptions.RequestException as e:
            logger.warning(f"NASA FIRMS API connection error: {e}")
            response_payload['message'] = f"NASA FIRMS connection error. Attempting open feed fallback."

    # 2. Fallback to NASA FIRMS Open NRT South Asia / Global Feeds
    if not fetch_success:
        open_feed_url = OPEN_NRT_FEEDS.get(source_key, OPEN_NRT_FEEDS["VIIRS_SNPP_NRT"])
        try:
            logger.info(f"Fetching from NASA FIRMS Open NRT Feed: {open_feed_url}")
            res = requests.get(open_feed_url, timeout=25, headers={'User-Agent': 'IndustrialThermalRiskSystem/1.0 (SIH Project)'})
            if res.status_code == 200 and len(res.text.strip()) > 50:
                raw_df = pd.read_csv(io.StringIO(res.text))
                fetch_success = True
                response_payload['is_live'] = True
                if not api_key:
                    response_payload['message'] = (
                        "Connected to NASA FIRMS Open Near-Real-Time Feed (South Asia 24h). "
                        "Add personal FIRMS_API_KEY for custom multi-day / global queries."
                    )
                else:
                    response_payload['message'] = "Connected via NASA FIRMS Open NRT Feed fallback."
        except requests.exceptions.RequestException as ex:
            logger.warning(f"NASA FIRMS Open Feed also failed: {ex}")

    # 3. Fallback to Disk Cache if live API failed
    if not fetch_success or raw_df is None or raw_df.empty:
        cached_df, cached_time = load_last_valid_cache()
        if not cached_df.empty:
            raw_df = cached_df
            response_payload['is_cached'] = True
            response_payload['message'] = f"NASA FIRMS is temporarily unreachable. Displaying last successfully cached dataset (from {cached_time})."
            logger.info("Loaded backup FIRMS dataset from disk cache.")
        else:
            response_payload['success'] = False
            response_payload['message'] = (
                "NASA FIRMS is currently unreachable and no cached data is available. "
                "Switch to 'Demo / Practice Dataset' mode or check your internet connection."
            )
            return response_payload

    # 4. Data Validation and Normalization
    clean_df, diag = validate_fire_dataframe(raw_df)
    response_payload['diagnostics'] = diag

    if clean_df.empty:
        response_payload['success'] = False
        response_payload['message'] = "No valid thermal observations found in the returned dataset."
        return response_payload

    response_payload['success'] = True
    response_payload['data'] = clean_df

    # Save to disk cache if this was a fresh live fetch
    if response_payload['is_live']:
        save_last_valid_cache(clean_df, source_key)

    return response_payload

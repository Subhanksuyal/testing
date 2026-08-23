import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.spatial import cKDTree

try:
    from .utils import calculate_distance_km
except (ImportError, ValueError):
    from utils import calculate_distance_km

logger = logging.getLogger("ThermalRiskIntelligence")

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]

# Memory cache for spatial clusters
_MEMORY_OSM_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_cache_dir() -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _query_overpass_with_failover(query: str, timeout_sec: int = 15) -> Optional[Dict[str, Any]]:
    """
    Executes an Overpass QL query with server failover and exponential retry.
    """
    for server in OVERPASS_SERVERS:
        for attempt in range(2):
            try:
                response = requests.post(
                    server,
                    data={'data': query},
                    timeout=timeout_sec,
                    headers={'User-Agent': 'IndustrialThermalRiskSystem/1.0 (SIH Project; Disaster Mitigation)'}
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'elements' in data:
                        return data
                elif response.status_code in [429, 502, 503, 504]:
                    logger.warning(f"Overpass server {server} returned {response.status_code}. Retrying...")
                    time.sleep(1.0 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.warning(f"Overpass server {server} attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
    return None


def fetch_osm_single_point(lat: float, lon: float, radius_km: float = 10.0) -> Dict[str, Any]:
    """
    Fetches nearby industrial and forest features for a single location.
    Used for the Single Observation Inspector.
    """
    cache_key = f"single_{round(lat, 2)}_{round(lon, 2)}_{round(radius_km, 1)}"
    if cache_key in _MEMORY_OSM_CACHE:
        return _MEMORY_OSM_CACHE[cache_key]

    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:12];
    (
      nwr["landuse"="industrial"](around:{radius_m},{lat},{lon});
      nwr["power"="plant"](around:{radius_m},{lat},{lon});
      nwr["industrial"](around:{radius_m},{lat},{lon});
      nwr["landuse"="forest"](around:{radius_m},{lat},{lon});
    );
    out center 40;
    """

    data = _query_overpass_with_failover(query, timeout_sec=12)
    result = {
        'success': False,
        'message': 'OpenStreetMap enrichment temporarily unavailable. Using ML baseline.',
        'nearest_industry_distance_km': None,
        'nearest_forest_distance_km': None,
        'nearest_industry_name': 'None found within radius',
        'nearest_industry_type': 'N/A',
        'industrial_features': [],
        'forest_features': []
    }

    if not data or 'elements' not in data:
        return result

    industrial_list = []
    forest_list = []

    for el in data['elements']:
        f_lat = el.get('lat') or el.get('center', {}).get('lat')
        f_lon = el.get('lon') or el.get('center', {}).get('lon')
        tags = el.get('tags', {})

        if f_lat is None or f_lon is None:
            continue

        dist = calculate_distance_km(lat, lon, f_lat, f_lon)
        if dist is None:
            continue

        if tags.get('landuse') == 'industrial' or 'power' in tags or 'industrial' in tags:
            name = tags.get('name', tags.get('operator', 'Industrial Facility'))
            ind_type = tags.get('industrial', tags.get('power', tags.get('landuse', 'industrial')))
            industrial_list.append({
                'name': name,
                'type': ind_type,
                'lat': f_lat,
                'lon': f_lon,
                'distance': round(dist, 2)
            })
        elif tags.get('landuse') == 'forest' or tags.get('natural') == 'wood':
            forest_list.append({
                'name': tags.get('name', 'Forest Area'),
                'type': 'forest',
                'lat': f_lat,
                'lon': f_lon,
                'distance': round(dist, 2)
            })

    industrial_list.sort(key=lambda x: x['distance'])
    forest_list.sort(key=lambda x: x['distance'])

    result['success'] = True
    result['message'] = f"Found {len(industrial_list)} industrial and {len(forest_list)} forest features."
    if industrial_list:
        result['nearest_industry_distance_km'] = industrial_list[0]['distance']
        result['nearest_industry_name'] = industrial_list[0]['name']
        result['nearest_industry_type'] = industrial_list[0]['type']
    if forest_list:
        result['nearest_forest_distance_km'] = forest_list[0]['distance']
    result['industrial_features'] = industrial_list[:10]
    result['forest_features'] = forest_list[:10]

    _MEMORY_OSM_CACHE[cache_key] = result
    return result


def enrich_dataframe_with_osm(
    df: pd.DataFrame,
    search_radius_km: float = 30.0,
    max_clusters: int = 15
) -> Tuple[pd.DataFrame, str]:
    """
    High-performance GIS batch enrichment.
    1. Bounding box query for the observations.
    2. Constructs KDTree for ultra-fast spatial search in milliseconds.
    3. Graceful fallback if Overpass fails.
    """
    df_enriched = df.copy()

    # If distances are already populated and valid, return
    if 'industry_distance_km' in df_enriched.columns and 'forest_distance_km' in df_enriched.columns:
        if df_enriched['industry_distance_km'].notna().any() and df_enriched['forest_distance_km'].notna().any():
            return df_enriched, "Using pre-calculated GIS distance features."

    if df_enriched.empty or 'latitude' not in df_enriched.columns or 'longitude' not in df_enriched.columns:
        df_enriched['industry_distance_km'] = np.nan
        df_enriched['forest_distance_km'] = np.nan
        return df_enriched, "No coordinates available for OSM enrichment."

    min_lat, max_lat = df_enriched['latitude'].min(), df_enriched['latitude'].max()
    min_lon, max_lon = df_enriched['longitude'].min(), df_enriched['longitude'].max()

    # Expand bounding box slightly for margin (~0.25 deg ~= 28km)
    pad = search_radius_km / 111.0
    s, n = max(-90.0, min_lat - pad), min(90.0, max_lat + pad)
    w, e = max(-180.0, min_lon - pad), min(180.0, max_lon + pad)

    # Check disk cache for this region
    cache_key = f"bbox_{round(s, 2)}_{round(w, 2)}_{round(n, 2)}_{round(e, 2)}"
    cache_file = os.path.join(_get_cache_dir(), f"{cache_key}.json")

    osm_elements = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                osm_elements = json.load(f)
                logger.info(f"Loaded {len(osm_elements)} OSM elements from disk cache {cache_key}")
        except Exception:
            osm_elements = None

    if osm_elements is None:
        # Bounded Overpass query
        query = f"""
        [out:json][timeout:20];
        (
          nwr["landuse"="industrial"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});
          nwr["power"="plant"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});
          nwr["industrial"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});
          nwr["landuse"="forest"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});
        );
        out center 2000;
        """
        data = _query_overpass_with_failover(query, timeout_sec=20)
        if data and 'elements' in data:
            osm_elements = data['elements']
            try:
                with open(cache_file, 'w') as f:
                    json.dump(osm_elements, f)
            except Exception as ex:
                logger.warning(f"Could not write OSM cache: {ex}")

    if not osm_elements:
        logger.warning("OSM Overpass API unreachable or empty. Gracefully marking GIS features as NaN.")
        if 'industry_distance_km' not in df_enriched.columns:
            df_enriched['industry_distance_km'] = np.nan
        if 'forest_distance_km' not in df_enriched.columns:
            df_enriched['forest_distance_km'] = np.nan
        return df_enriched, "OpenStreetMap enrichment temporarily unavailable. Proceeding with ML fallback."

    ind_coords = []
    forest_coords = []

    for el in osm_elements:
        f_lat = el.get('lat') or el.get('center', {}).get('lat')
        f_lon = el.get('lon') or el.get('center', {}).get('lon')
        tags = el.get('tags', {})

        if f_lat is None or f_lon is None:
            continue

        if tags.get('landuse') == 'industrial' or 'power' in tags or 'industrial' in tags:
            ind_coords.append((f_lat, f_lon))
        elif tags.get('landuse') == 'forest' or tags.get('natural') == 'wood':
            forest_coords.append((f_lat, f_lon))

    # Fast spatial indexing using cKDTree in radian metric (converted to km)
    # Earth radius R = 6371.0 km
    R = 6371.0
    fire_lat_rad = np.radians(df_enriched['latitude'].values)
    fire_lon_rad = np.radians(df_enriched['longitude'].values)
    # Convert lat/lon radians to 3D Cartesian on unit sphere for exact Euclidean/Chord distance in cKDTree
    fire_x = np.cos(fire_lat_rad) * np.cos(fire_lon_rad)
    fire_y = np.cos(fire_lat_rad) * np.sin(fire_lon_rad)
    fire_z = np.sin(fire_lat_rad)
    fire_pts_3d = np.column_stack([fire_x, fire_y, fire_z])

    # 1. Industrial nearest distance
    if ind_coords:
        ind_lats = np.radians([p[0] for p in ind_coords])
        ind_lons = np.radians([p[1] for p in ind_coords])
        ix = np.cos(ind_lats) * np.cos(ind_lons)
        iy = np.cos(ind_lats) * np.sin(ind_lons)
        iz = np.sin(ind_lats)
        ind_tree = cKDTree(np.column_stack([ix, iy, iz]))
        
        chord_dists, _ = ind_tree.query(fire_pts_3d, k=1)
        # Convert chord distance to great circle arc distance: d = 2 * R * arcsin(chord / 2)
        arc_dists = 2.0 * R * np.arcsin(np.clip(chord_dists / 2.0, 0.0, 1.0))
        df_enriched['industry_distance_km'] = np.round(arc_dists, 2)
    else:
        df_enriched['industry_distance_km'] = np.nan

    # 2. Forest nearest distance
    if forest_coords:
        for_lats = np.radians([p[0] for p in forest_coords])
        for_lons = np.radians([p[1] for p in forest_coords])
        fx = np.cos(for_lats) * np.cos(for_lons)
        fy = np.cos(for_lats) * np.sin(for_lons)
        fz = np.sin(for_lats)
        forest_tree = cKDTree(np.column_stack([fx, fy, fz]))
        
        chord_dists, _ = forest_tree.query(fire_pts_3d, k=1)
        arc_dists = 2.0 * R * np.arcsin(np.clip(chord_dists / 2.0, 0.0, 1.0))
        df_enriched['forest_distance_km'] = np.round(arc_dists, 2)
    else:
        df_enriched['forest_distance_km'] = np.nan

    msg = f"Enriched with OSM ({len(ind_coords)} industrial & {len(forest_coords)} forest features)."
    return df_enriched, msg

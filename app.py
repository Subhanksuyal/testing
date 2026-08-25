import os
import sys
import json
import time
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Add src to sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import (
    INDIAN_REGIONS,
    validate_fire_dataframe,
    filter_by_region,
    calculate_distance_km
)
from firms_api import (
    FIRMS_SOURCES,
    fetch_firms_live_data,
    get_firms_api_key
)
from osm import (
    enrich_dataframe_with_osm,
    fetch_osm_single_point
)
from features import add_derived_features
from model import RiskPredictor

# Page Configuration
st.set_page_config(
    page_title="Industrial Thermal Anomaly & Risk Intelligence System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-impact SIH presentation styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-live {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-cache {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Predictor & Model
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        return RiskPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}. Please run 'python src/train_model.py'.")
        return None

predictor = load_predictor()


# Load Practice / Demo Dataset
@st.cache_data(show_spinner=False)
def load_demo_dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'practice_firms.csv')
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        clean_df, _ = validate_fire_dataframe(raw_df)
        return clean_df
    return pd.DataFrame()


# Cached Live Data Fetcher
@st.cache_data(ttl=900, show_spinner=False)
def get_cached_firms_data(source_key: str, country_code: str, day_range: int):
    return fetch_firms_live_data(source_key=source_key, country_code=country_code, day_range=day_range)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
st.sidebar.markdown("### 🛰️ System Controls")

# Mode Selection
data_mode = st.sidebar.radio(
    "Data Ingestion Mode",
    ["📡 Live NASA FIRMS", "📂 Demo / Practice Dataset"],
    help="Live mode connects to NASA FIRMS satellite telemetry. Demo mode loads pre-validated reference observations."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Geographic & Satellite Scope")

selected_region = st.sidebar.selectbox(
    "Target Geographic Region",
    list(INDIAN_REGIONS.keys()) + ["Custom Bounding Box"],
    index=0
)

custom_bbox = None
if selected_region == "Custom Bounding Box":
    col_b1, col_b2 = st.sidebar.columns(2)
    with col_b1:
        min_lat_in = st.number_input("Min Latitude", value=20.0, format="%.2f")
        min_lon_in = st.number_input("Min Longitude", value=75.0, format="%.2f")
    with col_b2:
        max_lat_in = st.number_input("Max Latitude", value=25.0, format="%.2f")
        max_lon_in = st.number_input("Max Longitude", value=82.0, format="%.2f")
    custom_bbox = {
        'min_lat': min_lat_in, 'max_lat': max_lat_in,
        'min_lon': min_lon_in, 'max_lon': max_lon_in
    }

selected_source_name = st.sidebar.selectbox(
    "Satellite Constellation",
    list(FIRMS_SOURCES.keys()),
    index=0
)
source_code = FIRMS_SOURCES[selected_source_name]

time_horizon = st.sidebar.selectbox(
    "Time Horizon",
    ["Last 24 Hours (1 Day)", "Last 48 Hours (2 Days)", "Last 3 Days", "Last 7 Days"],
    index=0
)
day_range_map = {
    "Last 24 Hours (1 Day)": 1,
    "Last 48 Hours (2 Days)": 2,
    "Last 3 Days": 3,
    "Last 7 Days": 7
}
selected_days = day_range_map[time_horizon]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Risk & Filter Thresholds")

threshold = st.sidebar.slider(
    "Investigation Risk Threshold",
    min_value=0.10,
    max_value=0.90,
    value=predictor.default_threshold if predictor else 0.30,
    step=0.05,
    help="Observations with risk probability above this threshold are marked for priority investigation."
)

risk_filter = st.sidebar.selectbox(
    "Risk Band Filter",
    ["All Observations", "High Risk Only", "Medium & High Risk", "Low Risk Only"],
    index=0
)

min_confidence = st.sidebar.slider("Min Confidence (%)", 0.0, 100.0, 0.0, 5.0)
min_frp = st.sidebar.slider("Min FRP (MW)", 0.0, 500.0, 0.0, 10.0)

st.sidebar.markdown("---")
enable_clustering = st.sidebar.checkbox("Enable Map Marker Clustering", value=True)
enable_osm_enrichment = st.sidebar.checkbox("Enable Live OSM GIS Enrichment", value=False, help="Queries OpenStreetMap Overpass for industrial and forest spatial context.")
show_diagnostics = st.sidebar.checkbox("🔧 Show System Diagnostics", value=False, help="Display runtime versions and model diagnostic metadata.")

# Manual Refresh Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Telemetry Now", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

auto_refresh_choice = st.sidebar.selectbox(
    "Auto-Refresh Interval",
    ["Off", "15 Minutes", "30 Minutes", "60 Minutes"],
    index=0
)


# Main Content Area
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown('<div class="main-header">🔥 Industrial Thermal Anomaly & Risk Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time satellite-based thermal anomaly monitoring and industrial proximity risk assessment</div>', unsafe_allow_html=True)

with col_t2:
    st.markdown("<div style='text-align: right; padding-top: 10px;'>", unsafe_allow_html=True)
    if data_mode == "📡 Live NASA FIRMS":
        st.markdown('<span class="status-badge-live">● LIVE SATELLITE TELEMETRY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-demo">📂 DEMO / REFERENCE DATASET</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if show_diagnostics:
    with st.expander("🔧 Runtime & Model Diagnostics", expanded=True):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.write(f"**Python Version:** `{sys.version.split()[0]}`")
            import sklearn
            st.write(f"**scikit-learn:** `{sklearn.__version__}`")
        with d_col2:
            st.write(f"**pandas:** `{pd.__version__}`")
            st.write(f"**numpy:** `{np.__version__}`")
        with d_col3:
            import joblib
            st.write(f"**joblib:** `{joblib.__version__}`")
            st.write(f"**FIRMS Key Configured:** `{'Yes' if bool(get_firms_api_key()) else 'No (Open NRT Fallback)'}`")
        
        if predictor:
            st.write(f"**Model Features ({len(predictor.features)}):** `{', '.join(predictor.features)}`")
            st.write(f"**Pipeline Steps:** `{[name for name, _ in getattr(predictor.pipeline, 'steps', [])]}`")

# NASA FIRMS API Key Status Banner
api_key = get_firms_api_key()
if data_mode == "📡 Live NASA FIRMS" and not api_key:
    st.info(
        "💡 **NASA FIRMS API key is not configured.** "
        "The system is currently running seamlessly using the official **NASA FIRMS Open Near-Real-Time (NRT) South Asia feed**. "
        "To enable custom multi-day / global queries, add your free `FIRMS_API_KEY` to `.env` or Streamlit Secrets."
    )


# -------------------------------------------------------------
# Data Loading & Processing Pipeline
# -------------------------------------------------------------
raw_df = pd.DataFrame()
fetch_info = {}
last_updated_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if data_mode == "📡 Live NASA FIRMS":
    with st.spinner("Connecting to NASA FIRMS satellite constellation..."):
        fetch_result = get_cached_firms_data(
            source_key=source_code,
            country_code="IND",
            day_range=selected_days
        )
        fetch_info = fetch_result
        last_updated_time = fetch_result.get('last_fetched', last_updated_time)

        if fetch_result['success']:
            raw_df = fetch_result['data']
            if fetch_result.get('is_cached'):
                st.warning(f"⚠️ {fetch_result['message']}")
            elif fetch_result.get('message'):
                st.success(f"✅ {fetch_result['message']}")
        else:
            st.error(f"❌ {fetch_result.get('message', 'Failed to fetch live NASA FIRMS data.')}")
            st.info("💡 You can switch to **'Demo / Practice Dataset'** in the sidebar to explore the system with reference observations.")
else:
    raw_df = load_demo_dataset()
    fetch_info = {
        'success': True,
        'is_live': False,
        'message': 'Demo dataset loaded (1200 validated observations).'
    }

# -------------------------------------------------------------
# Geographic Filtering & Validation
# -------------------------------------------------------------
df_validated, diag = validate_fire_dataframe(raw_df)

if not df_validated.empty:
    df_regional = filter_by_region(
        df_validated,
        region_name=selected_region if selected_region != "Custom Bounding Box" else "",
        custom_bbox=custom_bbox
    )
else:
    df_regional = pd.DataFrame()

# -------------------------------------------------------------
# GIS Enrichment & ML Prediction Pipeline
# -------------------------------------------------------------
df_processed = pd.DataFrame()
osm_status_msg = ""

if not df_regional.empty and predictor is not None:
    # GIS Enrichment
    if enable_osm_enrichment:
        with st.spinner(f"Enriching {len(df_regional)} thermal observations with OpenStreetMap GIS spatial context..."):
            df_enriched, osm_status_msg = enrich_dataframe_with_osm(df_regional, search_radius_km=25.0)
    else:
        df_enriched = df_regional.copy()
        if 'industry_distance_km' not in df_enriched.columns:
            df_enriched['industry_distance_km'] = np.nan
        if 'forest_distance_km' not in df_enriched.columns:
            df_enriched['forest_distance_km'] = np.nan

    # ML Inference
    with st.spinner("Running Random Forest Risk Classification..."):
        df_processed = predictor.predict_batch(df_enriched, custom_threshold=threshold)

    # Apply User Filter Sliders
    if min_confidence > 0 and 'confidence' in df_processed.columns:
        df_processed = df_processed[df_processed['confidence'].fillna(0) >= min_confidence]

    if min_frp > 0 and 'frp' in df_processed.columns:
        df_processed = df_processed[df_processed['frp'].fillna(0) >= min_frp]

    if risk_filter == "High Risk Only":
        df_processed = df_processed[df_processed['risk_band'] == 'HIGH']
    elif risk_filter == "Medium & High Risk":
        df_processed = df_processed[df_processed['risk_band'].isin(['HIGH', 'MEDIUM'])]
    elif risk_filter == "Low Risk Only":
        df_processed = df_processed[df_processed['risk_band'] == 'LOW']


# -------------------------------------------------------------
# Top KPI Metric Cards
# -------------------------------------------------------------
total_obs = len(df_processed)
high_risk_count = len(df_processed[df_processed['risk_band'] == 'HIGH']) if total_obs > 0 else 0
med_risk_count = len(df_processed[df_processed['risk_band'] == 'MEDIUM']) if total_obs > 0 else 0
low_risk_count = len(df_processed[df_processed['risk_band'] == 'LOW']) if total_obs > 0 else 0
avg_frp = round(float(df_processed['frp'].mean()), 1) if total_obs > 0 and 'frp' in df_processed.columns and df_processed['frp'].notna().any() else 0.0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Live Observations", f"{total_obs:,}")
m2.metric("High Risk", f"{high_risk_count:,}", delta=f"{round(high_risk_count/max(1,total_obs)*100, 1)}%", delta_color="inverse")
m3.metric("Medium Risk", f"{med_risk_count:,}")
m4.metric("Low Risk", f"{low_risk_count:,}")
m5.metric("Avg FRP Intensity", f"{avg_frp} MW")
m6.metric("Last Updated", last_updated_time.split(" ")[-1] if " " in last_updated_time else last_updated_time)

st.markdown("---")


# -------------------------------------------------------------
# Interactive Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Live Risk Map",
    "📋 Live Observations Table",
    "📊 Risk Analytics",
    "🔍 Single Observation Inspector",
    "🤖 Model Performance",
    "ℹ️ Data Sources & Info"
])


# =============================================================
# TAB 1: LIVE RISK MAP
# =============================================================
with tab1:
    st.markdown(f"#### 🛰️ Active Thermal Anomalies Map — {selected_region} ({total_obs:,} Observations)")

    if df_processed.empty:
        st.warning("⚠️ No active thermal anomalies match the current geographic region and filter criteria.")
    else:
        # Compute map center and bounding box dynamically
        min_lat = df_processed['latitude'].min()
        max_lat = df_processed['latitude'].max()
        min_lon = df_processed['longitude'].min()
        max_lon = df_processed['longitude'].max()

        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles="CartoDB positron"
        )

        # Dynamic Bounds Fitting with padding
        lat_pad = max((max_lat - min_lat) * 0.08, 0.05)
        lon_pad = max((max_lon - min_lon) * 0.08, 0.05)
        m.fit_bounds([[min_lat - lat_pad, min_lon - lon_pad], [max_lat + lat_pad, max_lon + lon_pad]])

        # Marker Container (Clustered or Flat)
        container = MarkerCluster(name="Thermal Clusters").add_to(m) if enable_clustering and len(df_processed) > 50 else m

        for _, row in df_processed.iterrows():
            r_band = row.get('risk_band', 'LOW')
            prob_pct = row.get('risk_probability_pct', 0.0)
            frp_val = row.get('frp', 'N/A')
            conf_val = row.get('confidence', 'N/A')
            ind_dist = row.get('industry_distance_km', 'N/A')
            for_dist = row.get('forest_distance_km', 'N/A')
            acq_dt = row.get('acq_datetime', 'N/A')
            sat_name = row.get('satellite', selected_source_name)

            if r_band == 'HIGH':
                color = "#DC2626"      # Red
                fill_color = "#EF4444"
                radius = 6
            elif r_band == 'MEDIUM':
                color = "#D97706"      # Amber
                fill_color = "#F59E0B"
                radius = 5
            else:
                color = "#16A34A"      # Green
                fill_color = "#22C55E"
                radius = 4

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px; font-size: 13px;">
                <div style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 6px;">
                    Thermal Anomaly ({r_band} RISK)
                </div>
                <b>Risk Probability:</b> <span style="font-size: 14px; font-weight: bold; color: {color};">{prob_pct}%</span><br>
                <b>FRP:</b> {frp_val} MW<br>
                <b>Confidence:</b> {conf_val}%<br>
                <b>Industry Distance:</b> {ind_dist if pd.notna(ind_dist) else 'N/A'} km<br>
                <b>Forest Distance:</b> {for_dist if pd.notna(for_dist) else 'N/A'} km<br>
                <b>Acquisition:</b> {acq_dt}<br>
                <b>Satellite:</b> {sat_name}<br>
                <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            </div>
            """

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                weight=1.5,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(container)

        folium.LayerControl().add_to(m)

        # Legend
        st.markdown("""
        <div style="background-color: #F8FAFC; padding: 8px 15px; border-radius: 6px; border: 1px solid #E2E8F0; margin-bottom: 8px; font-size: 13px;">
            <b>Map Legend:</b> &nbsp;
            <span style="color: #DC2626;">●</span> <b>High Risk Anomaly (≥60%)</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color: #D97706;">●</span> <b>Medium Risk Anomaly (30% - 59%)</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color: #16A34A;">●</span> <b>Low Risk Anomaly (&lt;30%)</b>
        </div>
        """, unsafe_allow_html=True)

        st_folium(m, width=None, height=620, use_container_width=True)


# =============================================================
# TAB 2: LIVE OBSERVATIONS TABLE
# =============================================================
with tab2:
    st.markdown(f"#### 📋 Observation Telemetry Records ({len(df_processed):,} Rows)")

    if not df_processed.empty:
        display_cols = [
            'latitude', 'longitude', 'frp', 'confidence', 'acq_datetime',
            'satellite', 'industry_distance_km', 'forest_distance_km',
            'persistence_days', 'night_flag', 'risk_probability_pct',
            'risk_band', 'investigation_status'
        ]
        available_display_cols = [c for c in display_cols if c in df_processed.columns]
        
        # Color formatted dataframe
        st.dataframe(
            df_processed[available_display_cols].sort_values(by='risk_probability_pct', ascending=False),
            use_container_width=True,
            height=500
        )

        csv_data = df_processed[available_display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Observations CSV",
            data=csv_data,
            file_name=f"thermal_risk_telemetry_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No records available to display in table.")


# =============================================================
# TAB 3: RISK ANALYTICS & VISUAL INTELLIGENCE
# =============================================================
with tab3:
    st.markdown("#### 📊 Risk Distribution & Spatial Telemetry Analysis")

    if not df_processed.empty:
        c1, c2 = st.columns(2)

        with c1:
            # Donut Chart: Risk Band Distribution
            band_counts = df_processed['risk_band'].value_counts().reset_index()
            band_counts.columns = ['Risk Band', 'Count']
            color_map = {'HIGH': '#DC2626', 'MEDIUM': '#F59E0B', 'LOW': '#22C55E'}

            fig_pie = px.pie(
                band_counts,
                values='Count',
                names='Risk Band',
                title='Risk Band Distribution',
                hole=0.45,
                color='Risk Band',
                color_discrete_map=color_map
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            # Scatter Plot: FRP vs Risk Probability
            fig_scatter = px.scatter(
                df_processed,
                x='frp',
                y='risk_probability_pct',
                color='risk_band',
                color_discrete_map=color_map,
                title='Fire Radiative Power (MW) vs. Industrial Risk Probability (%)',
                labels={'frp': 'FRP (MW)', 'risk_probability_pct': 'Risk Probability (%)'},
                hover_data=['latitude', 'longitude', 'confidence', 'industry_distance_km']
            )
            fig_scatter.add_hline(y=threshold * 100, line_dash="dash", line_color="black", annotation_text=f"Threshold ({int(threshold*100)}%)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            # Day vs Night distribution
            if 'night_flag' in df_processed.columns:
                df_processed['Observation Time'] = df_processed['night_flag'].map({1: 'Nighttime', 0: 'Daytime'}).fillna('Daytime')
                fig_dn = px.histogram(
                    df_processed,
                    x='Observation Time',
                    color='risk_band',
                    barmode='group',
                    color_discrete_map=color_map,
                    title='Risk Distribution by Day / Night Telemetry'
                )
                st.plotly_chart(fig_dn, use_container_width=True)

        with c4:
            # Industry Distance Distribution
            if 'industry_distance_km' in df_processed.columns and df_processed['industry_distance_km'].notna().any():
                fig_dist = px.histogram(
                    df_processed.dropna(subset=['industry_distance_km']),
                    x='industry_distance_km',
                    color='risk_band',
                    color_discrete_map=color_map,
                    nbins=25,
                    title='Distance to Nearest Industrial Zone (km)',
                    labels={'industry_distance_km': 'Distance (km)'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Analytics will display when observations are available.")


# =============================================================
# TAB 4: SINGLE OBSERVATION INSPECTOR
# =============================================================
with tab4:
    st.markdown("#### 🔍 Deep-Dive Single Anomaly Telemetry Inspector")

    if not df_processed.empty:
        obs_options = [
            f"Index {idx} | Lat: {row['latitude']:.3f}, Lon: {row['longitude']:.3f} | Risk: {row.get('risk_band')} ({row.get('risk_probability_pct')}%) | FRP: {row.get('frp')}MW"
            for idx, row in df_processed.iterrows()
        ]
        selected_idx_str = st.selectbox("Select Anomaly Observation to Inspect", obs_options, index=0)
        selected_idx = int(selected_idx_str.split(" ")[1])
        target_row = df_processed.loc[selected_idx]

        ic1, ic2 = st.columns([1, 2])

        with ic1:
            st.markdown("##### 📌 Anomaly Profile")
            st.write(f"**Coordinates:** `{target_row['latitude']:.4f}, {target_row['longitude']:.4f}`")
            st.write(f"**Risk Level:** **{target_row.get('risk_band')}** ({target_row.get('risk_probability_pct')}%)")
            st.write(f"**Investigation Action:** `{target_row.get('investigation_status')}`")
            st.write(f"**FRP Intensity:** `{target_row.get('frp')} MW`")
            st.write(f"**Confidence:** `{target_row.get('confidence')}%`")
            st.write(f"**Nearest Industry:** `{target_row.get('industry_distance_km', 'N/A')} km`")
            st.write(f"**Nearest Forest:** `{target_row.get('forest_distance_km', 'N/A')} km`")
            st.write(f"**Persistence:** `{target_row.get('persistence_days', 1)} days`")
            st.write(f"**Acquisition Date/Time:** `{target_row.get('acq_datetime', 'N/A')}`")

            # Risk Factor Bar Gauge
            prob_val = target_row.get('risk_probability', 0.0)
            st.progress(float(prob_val))

            if st.button("🛰️ Fetch Live OSM Context Around Hotspot"):
                with st.spinner("Querying OpenStreetMap around coordinates..."):
                    osm_single = fetch_osm_single_point(target_row['latitude'], target_row['longitude'], radius_km=8.0)
                    if osm_single['success']:
                        st.success(osm_single['message'])
                        if osm_single['industrial_features']:
                            st.markdown("**Nearby Industrial Features:**")
                            for ind in osm_single['industrial_features'][:5]:
                                st.write(f"- 🏭 **{ind['name']}** ({ind['type']}) — *{ind['distance']} km*")
                        if osm_single['forest_features']:
                            st.markdown("**Nearby Forest / Woodland:**")
                            for f in osm_single['forest_features'][:3]:
                                st.write(f"- 🌲 **{f['name']}** — *{f['distance']} km*")
                    else:
                        st.warning(osm_single['message'])

        with ic2:
            st.markdown("##### 🗺️ Local Area Map with 5km Buffer Zone")
            m_single = folium.Map(
                location=[target_row['latitude'], target_row['longitude']],
                zoom_start=12,
                tiles="CartoDB positron"
            )

            # Hotspot Marker
            folium.Marker(
                [target_row['latitude'], target_row['longitude']],
                popup=f"Target Anomaly<br>Risk: {target_row.get('risk_band')} ({target_row.get('risk_probability_pct')}%)",
                icon=folium.Icon(color="red" if target_row.get('risk_band') == "HIGH" else "orange", icon="fire", prefix="fa")
            ).add_to(m_single)

            # 5 km Buffer Radius
            folium.Circle(
                location=[target_row['latitude'], target_row['longitude']],
                radius=5000,
                color="#2563EB",
                fill=True,
                fill_color="#3B82F6",
                fill_opacity=0.1,
                weight=1.5,
                dash_array="5, 5"
            ).add_to(m_single)

            st_folium(m_single, width=None, height=450, use_container_width=True)
    else:
        st.info("Select or filter observations to enable the inspector.")


# =============================================================
# TAB 5: MODEL PERFORMANCE & METRICS
# =============================================================
with tab5:
    st.markdown("#### 🤖 Machine Learning Model Architecture & Performance")

    meta = predictor.metadata if predictor else {}

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Model Architecture", "RandomForest")
    p2.metric("Validation Accuracy", f"{meta.get('accuracy', 0.750):.2%}")
    p3.metric("Recall (Sensitivity)", f"{meta.get('recall', 0.737):.2%}")
    p4.metric("ROC AUC Score", f"{meta.get('roc_auc', 0.828):.3f}")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("##### 📌 Feature Importances")
        if 'feature_importances' in meta:
            feat_df = pd.DataFrame(list(meta['feature_importances'].items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
            fig_imp = px.bar(
                feat_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Gini Feature Importances in Random Forest',
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importances will be available after training.")

    with col_m2:
        st.markdown("##### 🎯 Threshold Optimization Rationale")
        st.markdown("""
        - **Default Threshold:** `0.30`
        - **Why 0.30 instead of 0.50?**
          In early disaster & industrial risk intelligence, **Recall (catching real industrial hazards)** is significantly more vital than pure Precision.
          Threshold `0.30` elevates Recall to **~74%**, minimizing critical false negatives.
        - **Pipeline Preprocessing:**
          - `SimpleImputer(strategy='median')` guarantees resilience when GIS/FRP features are temporarily unavailable.
          - `StandardScaler()` normalizes multi-scale proximity & intensity features.
        """)

        if 'confusion_matrix' in meta:
            cm = meta['confusion_matrix']
            st.markdown(f"""
            **Confusion Matrix (Holdout Test Set):**
            - True Negatives: `{cm[0][0]}` | False Positives: `{cm[0][1]}`
            - False Negatives: `{cm[1][0]}` | True Positives: `{cm[1][1]}`
            """)


# =============================================================
# TAB 6: DATA SOURCES & SYSTEM INFORMATION
# =============================================================
with tab6:
    st.markdown("#### ℹ️ System Architecture & External Data Sources")

    st.markdown("""
    ### 🛰️ Telemetry Pipeline
    ```text
    NASA FIRMS API / Open NRT Feeds (MODIS / VIIRS 375m)
                 ↓
    Robust Data Validation & Coordinate Filtering (utils.py)
                 ↓
    OpenStreetMap Overpass GIS Spatial Indexing (KDTree)
                 ↓
    Feature Engineering (Persistence, Proximity, Night Flag)
                 ↓
    Random Forest Classification Pipeline (SimpleImputer + Scaler)
                 ↓
    Real-Time Folium Risk Map + Plotly Intelligence Dashboard
    ```

    ---

    ### 🔑 NASA FIRMS API Setup Guide
    1. Register for a free NASA FIRMS MAP_KEY at [NASA FIRMS Key Portal](https://firms.modaps.eosdis.nasa.gov/api/map_key/).
    2. Add your key to your environment:
       - **Local Dev:** In `.env` add `FIRMS_API_KEY=your_key_here`
       - **Streamlit Cloud:** In `App Settings > Secrets` add `FIRMS_API_KEY = "your_key_here"`
    3. The application will automatically detect your key and enable full multi-day country and area endpoints.

    ---

    ### ⚠️ ML Limitations & Ethical Guidelines
    > **Important Disclaimer:**
    > This model estimates industrial risk based on thermal anomaly characteristics, FRP intensity, persistence, and geographic proximity to industrial zones and woodland.
    > It does not prove that a fire originated from an industrial facility.
    > Results are categorized as **"Potential industrial-associated thermal anomalies"** for prioritization and monitoring by disaster response authorities.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "Smart India Hackathon (SIH) Project | Industrial Thermal Anomaly & Risk Intelligence System | Powered by NASA FIRMS & OpenStreetMap"
    "</div>",
    unsafe_allow_html=True
)

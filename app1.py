from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
import google.generativeai as genai
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import FastMarkerCluster, MarkerCluster
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import alphashape


# ================================================================ #
#  PAGE CONFIG — must be first Streamlit call                      #
# ================================================================ #

GEMINI_API_KEY = "API_KEY"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
import google.generativeai as genai

genai.configure(api_key="API_KEY")  # Replace with your actual Gemini API key
models = genai.list_models()
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)
model = genai.GenerativeModel("gemini-3-flash-preview")

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

st.markdown(
    """
    <section class="hero">
        <p class="hero-label">Groundwater Monitoring Platform</p>
        <h1 class="hero-title">Water Quality Intelligence Dashboard</h1>
        <p class="hero-subtitle">
            Live geospatial monitoring of station-level pH readings with hover-enabled location cards,
            district boundary context, and decision-friendly analytics.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Panel")
    selected_basemap = st.selectbox("Map Theme", list(BASEMAPS.keys()), index=0)

    min_ph = float(station_gdf["ph"].min())
    max_ph = float(station_gdf["ph"].max())
    ph_range = st.slider(
        "pH Filter",
        min_value=min_ph,
        max_value=max_ph,
        value=(min_ph, max_ph),
        step=0.01,
    )

    categories = st.multiselect(
        "Quality Class",
        options=CATEGORY_ORDER,
        default=CATEGORY_ORDER,
    )

    search_station = st.text_input("Find Station")

    # ---- COMPUTE filtered HERE (so it's available for the chat) ----
    filtered = station_gdf[
        (station_gdf["ph"] >= ph_range[0])
        & (station_gdf["ph"] <= ph_range[1])
        & (station_gdf["category"].isin(categories))
    ].copy()

    if search_station.strip():
        filtered = filtered[
            filtered["location"].str.contains(search_station.strip(), case=False, na=False)
        ].copy()

    # ---- AI ASSISTANT (now filtered is defined) ----
    st.divider()
    with st.expander("💬 AI Assistant", expanded=False):
        st.caption("Ask about water quality")
        # Display chat history
        for msg in st.session_state.gemini_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask Gemini..."):
            # Add user message
            st.session_state.gemini_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build context using filtered (defined above)
            context = build_chat_context(filtered)
            full_prompt = f"{context}\n\nUser question: {prompt}"
            try:
                response = model.generate_content(full_prompt)
                reply = response.text
            except Exception as e:
                reply = f"Error calling Gemini: {e}"

            # Add assistant message
            st.session_state.gemini_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

st.set_page_config(
    page_title="Water Quality Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# map_height after set_page_config so session_state is available
map_height = 420 if st.session_state.get("mobile", False) else 700

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BASEMAPS = {
    "Light":     {"tiles": "CartoDB positron", "attr": "CartoDB"},
    "Street":    {"tiles": "OpenStreetMap",    "attr": "OpenStreetMap"},
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr":  "Esri",
    },
}

CATEGORY_ORDER = ["Acidic", "Safe", "Alkaline"]
COLOR_MAP      = {"Acidic": "#ef4444", "Safe": "#22c55e", "Alkaline": "#2563eb"}


# ---------------- CSS ---------------- #

@st.cache_resource
def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


# ---------------- pH Logic ---------------- #

def classify_ph(ph):
    if ph < 6.5:  return "Acidic"
    if ph <= 8.5: return "Safe"
    return "Alkaline"


def marker_color(ph):
    if ph < 6.5:  return "#ef4444"
    if ph <= 8.5: return "#22c55e"
    return "#2563eb"


# ---------------- Metric Card ---------------- #

def metric_block(title, value, note):
    return (
        "<div class='metric-shell'>"
        f"<p class='metric-kicker'>{title}</p>"
        f"<p class='metric-number'>{value}</p>"
        f"<p class='metric-note'>{note}</p>"
        "</div>"
    )


# ---------------- Data Loader ---------------- #

@st.cache_resource(show_spinner="Loading district data…")
def load_data():
    boundary = gpd.read_file(DATA_DIR / "hisar_new_boundary.geojson").to_crs("EPSG:4326")
    villages = gpd.read_file(DATA_DIR / "hisar_final_village_boundary.geojson").to_crs("EPSG:4326")
    stations = gpd.read_file(DATA_DIR / "hisar_points_updated.geojson")

    stations_wgs84 = stations.to_crs("EPSG:4326")
    stations_3857  = stations.to_crs("EPSG:3857")

    location_candidates = ["Location", "location", "Village", "village", "Name", "name"]
    location_col = next((c for c in location_candidates if c in stations_wgs84.columns), None)
    ph_col       = next((c for c in stations_wgs84.columns if c.lower() == "ph"), None)

    stations_wgs84 = stations_wgs84.rename(
        columns={location_col: "location", ph_col: "ph"}
    ).copy()

    stations_wgs84["ph"]           = stations_wgs84["ph"].astype(float)
    stations_wgs84                 = stations_wgs84.dropna(subset=["geometry", "ph"])
    stations_wgs84["lat"]          = stations_wgs84.geometry.y
    stations_wgs84["lon"]          = stations_wgs84.geometry.x
    stations_wgs84["category"]     = stations_wgs84["ph"].apply(classify_ph)
    stations_wgs84["marker_color"] = stations_wgs84["ph"].apply(marker_color)

    return boundary, villages, stations_wgs84, stations_3857


# ---------------- Tooltip ---------------- #

def tooltip_html(row):
    return (
        f"<b>{row['location']}</b><br>"
        f"pH: {row['ph']:.2f}<br>"
        f"Category: {row['category']}<br>"
        f"Lat: {row['lat']:.4f}<br>"
        f"Lon: {row['lon']:.4f}"
    )


# ---------------- DBSCAN Clustering ---------------- #

def compute_clusters(stations):
    if stations.empty:
        return []
    coords = stations[["lat", "lon"]].values
    return DBSCAN(eps=0.02, min_samples=3).fit(coords).labels_


# ---------------- Map Builder ---------------- #

def build_map(boundary, villages, stations, center_lat, center_lon, basemap, show_heatmap=False):

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(**BASEMAPS[basemap]).add_to(m)

    folium.GeoJson(
        boundary,
        name="District Boundary",
        style_function=lambda _: {
            "fillColor": "#67e8f9",
            "fillOpacity": 0.1,
            "color": "#0f172a",
            "weight": 2,
        },
    ).add_to(m)

    folium.GeoJson(
        villages,
        name="Village Boundaries",
        style_function=lambda _: {
            "fillColor": "#00000000",
            "color": "#64748b",
            "weight": 1,
        },
        highlight_function=lambda _: {
            "color": "#38bdf8",
            "weight": 3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["vilname11"],
            aliases=["Village:"],
        ),
    ).add_to(m)

    # ---------- Spatial Clustering ---------- #

    stations = stations.copy()
    stations["cluster"] = compute_clusters(stations)

    # ---------- Marker Cluster ---------- #

    marker_cluster = MarkerCluster(
        name="Water Quality Stations",
        options={
            "zoomToBoundsOnClick":        True,
            "spiderfyOnMaxZoom":          True,
            "showCoverageOnHover":        False,
            "spiderfyDistanceMultiplier": 1.2,
        },
    ).add_to(m)

    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="#0f172a",
            weight=1,
            fill=True,
            fill_color=row["marker_color"],
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip_html(row)),
        ).add_to(marker_cluster)

    # ---------- Cluster Polygons ---------- #

    if len(stations) > 0 and "cluster" in stations.columns:
        for cluster_id in stations["cluster"].unique():
            if cluster_id == -1:
                continue

            cluster_points = stations[stations["cluster"] == cluster_id]
            coords_list = [(row["lon"], row["lat"]) for _, row in cluster_points.iterrows()]

            if len(coords_list) < 3:
                continue

            alpha_shape = alphashape.alphashape(coords_list, 0.5)
            avg_ph = cluster_points["ph"].mean()

            color = "#ef4444" if avg_ph < 6.5 else "#22c55e" if avg_ph <= 8.5 else "#2563eb"

            folium.GeoJson(
                alpha_shape,
                name=f"Cluster {cluster_id}",
                style_function=lambda _, col=color: {
                    "color": col, "fillColor": col,
                    "weight": 2, "fillOpacity": 0.18,
                },
                tooltip=f"Cluster {cluster_id} | Stations: {len(cluster_points)} | Avg pH: {avg_ph:.2f}",
            ).add_to(m)

    # ---------- Heatmap Layer (optional) ---------- #

    if show_heatmap and not stations.empty:
        from folium.plugins import HeatMap
        heat_data = [[row["lat"], row["lon"], row["ph"]] for _, row in stations.iterrows()]
        HeatMap(
            heat_data,
            name="pH Heatmap",
            min_opacity=0.3,
            radius=25,
            blur=20,
            gradient={0.3: "#22c55e", 0.6: "#eab308", 1.0: "#ef4444"},
        ).add_to(m)

    # ---------- Legend ---------- #

    legend_html = """
    <div class="map-legend">
        <div class="legend-title">pH Range</div>
        <div><span class="legend-dot acidic"></span>Acidic (&lt; 6.5)</div>
        <div><span class="legend-dot safe"></span>Safe (6.5 - 8.5)</div>
        <div><span class="legend-dot alkaline"></span>Alkaline (&gt; 8.5)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    return m


# ---------------- Nearest Station ---------------- #

def nearest_station(stations_3857, stations_4326, lat, lon):
    if stations_3857.empty:
        return None
    click_point = gpd.GeoSeries(
        [Point(lon, lat)], crs="EPSG:4326"
    ).to_crs("EPSG:3857").iloc[0]
    idx = stations_3857.geometry.distance(click_point).idxmin()
    return stations_4326.loc[idx]


# ================================================================ #
#  LOAD                                                            #
# ================================================================ #

load_css()
boundary_gdf, village_gdf, station_gdf, station_gdf_3857 = load_data()


# ---------------- Hero ---------------- #

st.markdown(
    """
    <section class="hero">
        <p class="hero-label">Groundwater Monitoring Platform</p>
        <h1 class="hero-title">Water Quality Intelligence Dashboard</h1>
        <p class="hero-subtitle">
            Live geospatial monitoring of station-level pH readings with interactive
            maps, village-level context, and actionable insights.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("Control Panel")

    selected_basemap = st.selectbox("Map Theme", list(BASEMAPS.keys()))

    mobile_layout = st.checkbox(
        "Mobile Layout",
        value=False,
        help="Stack map and insights vertically for small screens",
    )
    st.session_state["mobile"] = mobile_layout

    min_ph = float(station_gdf["ph"].min())
    max_ph = float(station_gdf["ph"].max())

    ph_range = st.slider(
        "pH Filter",
        min_value=min_ph,
        max_value=max_ph,
        value=(min_ph, max_ph),
        step=0.01,
    )

    categories = st.multiselect(
        "Quality Class",
        CATEGORY_ORDER,
        default=CATEGORY_ORDER,
    )

    station_list   = sorted(station_gdf["location"].dropna().unique())
    search_station = st.selectbox("Select Station", ["All Stations"] + station_list)

    st.markdown("---")
    st.subheader("Map Layers")
    show_heatmap = st.checkbox(
        "Show pH Heatmap",
        value=False,
        help="Kernel density heatmap of pH intensity across the district",
    )

    st.markdown("---")
    st.subheader("Download Data")
    download_filtered_placeholder = st.empty()
    download_full_placeholder     = st.empty()

    st.markdown("---")
    st.caption("📍 District: Hisar, Haryana")
    st.caption("🔬 Standard: BIS 10500:2012")


# ---------------- Filtering ---------------- #

filtered = station_gdf[
    (station_gdf["ph"] >= ph_range[0])
    & (station_gdf["ph"] <= ph_range[1])
    & (station_gdf["category"].isin(categories))
].copy()

if search_station != "All Stations":
    filtered = filtered[filtered["location"] == search_station]

filtered_3857 = station_gdf_3857.loc[filtered.index]


# ---------------- Download Buttons ---------------- #

csv_filtered = filtered[["location", "ph", "category", "lat", "lon"]].to_csv(index=False).encode("utf-8")
download_filtered_placeholder.download_button(
    label="Download Filtered Stations",
    data=csv_filtered,
    file_name="hisar_filtered_water_quality.csv",
    mime="text/csv",
)

csv_all = station_gdf[["location", "ph", "category", "lat", "lon"]].to_csv(index=False).encode("utf-8")
download_full_placeholder.download_button(
    label="Download Full Dataset",
    data=csv_all,
    file_name="hisar_all_stations.csv",
    mime="text/csv",
)


# ---------------- Metrics ---------------- #

active_count = len(filtered)
safe_count   = (filtered["category"] == "Safe").sum()
risk_count   = (filtered["category"] != "Safe").sum()
avg_ph       = filtered["ph"].mean() if not filtered.empty else 0

metric_cols = st.columns(4)
metric_cols[0].markdown(metric_block("Active Stations", active_count, "Filtered monitoring points"), unsafe_allow_html=True)
metric_cols[1].markdown(metric_block("Average pH",      f"{avg_ph:.2f}", "District snapshot"),       unsafe_allow_html=True)
metric_cols[2].markdown(metric_block("Safe Stations",   safe_count, "Safe water range"),             unsafe_allow_html=True)
metric_cols[3].markdown(metric_block("Attention Needed", risk_count, "Acidic or alkaline"),          unsafe_allow_html=True)


# ---------------- BIS Compliance Banner ---------------- #

if not filtered.empty:
    acidic       = (filtered["category"] == "Acidic").sum()
    alkaline     = (filtered["category"] == "Alkaline").sum()
    safe         = (filtered["category"] == "Safe").sum()
    safe_percent = (safe / len(filtered)) * 100

    if safe_percent >= 80:
        bar_color, icon, label = "#16a34a", "✅", "Good Standing"
    elif safe_percent >= 60:
        bar_color, icon, label = "#ca8a04", "⚠️", "Needs Attention"
    else:
        bar_color, icon, label = "#dc2626", "🔴", "Critical"

    st.markdown(
        f"""
        <div class="insight-card">
            <p class="insight-label">District Water Quality Summary</p>
            <p>
            {safe_percent:.0f}% of monitoring stations fall within the safe pH range (6.5–8.5).
            {acidic + alkaline} stations show non-neutral readings requiring attention.
            Average district pH is <strong>{avg_ph:.2f}</strong>.
            </p>
        </div>
        <div style="background:{bar_color}18; border-left:4px solid {bar_color};
                    padding:12px 16px; border-radius:6px; margin:10px 0 16px 0;">
            <strong>{icon} BIS 10500:2012 Compliance — {label}: {safe_percent:.1f}%</strong>
            &nbsp;|&nbsp;
            {int(len(filtered) - safe)} of {len(filtered)} stations exceed permissible pH limits
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Map + Insight Panel ---------------- #

center_lat = filtered["lat"].mean() if not filtered.empty else station_gdf["lat"].mean()
center_lon = filtered["lon"].mean() if not filtered.empty else station_gdf["lon"].mean()

is_mobile = st.session_state.get("mobile", False)
if is_mobile:
    map_col     = st.container()
    insight_col = st.container()
else:
    map_col, insight_col = st.columns([3.2, 1.35], gap="large")


with map_col:

    st.markdown("<h3 class='panel-heading'>Live Water Quality Map</h3>", unsafe_allow_html=True)

    m = build_map(
        boundary_gdf,
        village_gdf,
        filtered,
        center_lat,
        center_lon,
        selected_basemap,
        show_heatmap,
    )

    # MAP SCROLL FIX — MOBILE
    # .map-wrapper has touch-action:pan-y so the page scrolls freely
    # over the map. A CSS ::after shield intercepts all touches.
    # When session_state["map_interact"] is True, .map-active class
    # is added which removes the shield — Leaflet gets full control.
    # Button only rendered when Mobile Layout is ON — not shown on desktop at all.

    if "map_interact" not in st.session_state:
        st.session_state["map_interact"] = False

    is_interacting = st.session_state["map_interact"]
    shield_class   = "map-wrapper map-active" if is_interacting else "map-wrapper"

    st.markdown(f'<div class="{shield_class}">', unsafe_allow_html=True)
    map_state = st_folium(m, height=map_height, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Only render the button when the user has turned on Mobile Layout
    if is_mobile:
        btn_label = "✕ Done — scroll page" if is_interacting else "👆 Interact with Map"
        if st.button(btn_label, key="map_interact_real_btn", use_container_width=True):
            st.session_state["map_interact"] = not is_interacting
            st.rerun()


# ---------------- Station Insight Panel ---------------- #

with insight_col:

    st.markdown("<h3 class='panel-heading'>Station Insights</h3>", unsafe_allow_html=True)
    st.markdown('<div class="insight-scroll">', unsafe_allow_html=True)

    selected_station = None

    if map_state and map_state.get("last_clicked"):
        selected_station = nearest_station(
            filtered_3857,
            filtered,
            map_state["last_clicked"]["lat"],
            map_state["last_clicked"]["lng"],
        )

    if selected_station is None and not filtered.empty:
        selected_station = filtered.iloc[0]

    if selected_station is not None:

        difference = selected_station["ph"] - avg_ph
        status     = "⬆ Higher than avg" if difference > 0 else "⬇ Lower than avg"

        if selected_station["category"] == "Safe":
            badge_color, badge_text = "#16a34a", "✅ SAFE"
        elif selected_station["category"] == "Acidic":
            badge_color, badge_text = "#dc2626", "🔴 ACIDIC"
        else:
            badge_color, badge_text = "#2563eb", "🔵 ALKALINE"

        st.markdown(f"""
<div class="insight-card">
<p class="insight-label">Focused Station</p>
<h4>{selected_station['location']}</h4>
<span style="background:{badge_color}18; color:{badge_color};
             border:1px solid {badge_color}40; border-radius:4px;
             padding:2px 10px; font-size:0.78rem; font-weight:600;">
    {badge_text}
</span>
<hr style="margin:10px 0; border-color:#e2e8f0;">
<p><strong>pH:</strong> {selected_station['ph']:.2f}</p>
<p><strong>BIS Safe Range:</strong> 6.5 – 8.5</p>
<p><strong>District Avg:</strong> {avg_ph:.2f}</p>
<p><strong>Difference:</strong> {difference:+.2f} ({status})</p>
<hr style="margin:10px 0; border-color:#e2e8f0;">
<p><strong>Latitude:</strong> {selected_station['lat']:.5f}</p>
<p><strong>Longitude:</strong> {selected_station['lon']:.5f}</p>
</div>
""", unsafe_allow_html=True)

        st.progress(min(max(selected_station["ph"] / 14.0, 0), 1))

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(selected_station["ph"]),
            title={"text": "pH Level"},
            gauge={
                "axis": {"range": [0, 14]},
                "bar":  {"color": "#0ea5e9"},
                "steps": [
                    {"range": [0,   6.5], "color": "#d75d5d"},
                    {"range": [6.5, 8.5], "color": "#4dd17b"},
                    {"range": [8.5, 14],  "color": "#6596d3"},
                ],
                "threshold": {
                    "line":      {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value":     float(selected_station["ph"]),
                },
            },
        ))
        gauge.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge, use_container_width=True)

        percentile = (filtered["ph"] < selected_station["ph"]).mean() * 100
        st.markdown(
            f"""
            <div class="insight-card">
            <p class="insight-label">Station Ranking</p>
            <p>This station has higher pH than <strong>{percentile:.1f}%</strong> of monitoring stations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Analytics Tabs ---------------- #

st.markdown("<h3 class='panel-heading'>Water Quality Analytics</h3>", unsafe_allow_html=True)

if not filtered.empty:
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 pH Histogram", "📦 Box Plot"])

    with tab1:
        dist    = filtered["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
        dist_df = dist.reset_index()
        dist_df.columns = ["Category", "Count"]

        fig_bar = px.bar(
            dist_df, x="Category", y="Count",
            color="Category", color_discrete_map=COLOR_MAP, text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        fig_hist = px.histogram(
            filtered, x="ph", nbins=30,
            color="category", color_discrete_map=COLOR_MAP,
            labels={"ph": "pH Value", "count": "Stations"},
            category_orders={"category": CATEGORY_ORDER},
        )
        fig_hist.add_vline(
            x=6.5, line_dash="dash", line_color="#ef4444", line_width=2,
            annotation_text="BIS Min (6.5)", annotation_position="top right",
            annotation_font_color="#ef4444",
        )
        fig_hist.add_vline(
            x=8.5, line_dash="dash", line_color="#2563eb", line_width=2,
            annotation_text="BIS Max (8.5)", annotation_position="top left",
            annotation_font_color="#2563eb",
        )
        fig_hist.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=10, r=10, t=20, b=10), bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        fig_box = px.box(
            filtered, x="category", y="ph",
            color="category", color_discrete_map=COLOR_MAP,
            points="all", hover_name="location",
            category_orders={"category": CATEGORY_ORDER},
            labels={"ph": "pH Value", "category": "Category"},
        )
        fig_box.add_hline(y=6.5, line_dash="dot", line_color="#ef4444",
                          annotation_text="BIS Min", annotation_position="right")
        fig_box.add_hline(y=8.5, line_dash="dot", line_color="#2563eb",
                          annotation_text="BIS Max", annotation_position="right")
        fig_box.update_layout(
            template="plotly_white", height=340,
            showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ---------------- Worst Stations Table ---------------- #

st.markdown("<h3 class='panel-heading'>⚠️ Stations Needing Attention</h3>", unsafe_allow_html=True)

if not filtered.empty:
    worst = filtered[filtered["category"] != "Safe"].copy()

    if not worst.empty:
        worst["deviation"] = (worst["ph"] - 7.5).abs()
        worst_display = (
            worst.nlargest(5, "deviation")[["location", "ph", "category", "lat", "lon"]]
            .rename(columns={
                "location": "Station", "ph": "pH",
                "category": "Status", "lat": "Latitude", "lon": "Longitude",
            })
            .reset_index(drop=True)
        )

        def highlight_status(val):
            if val == "Acidic":   return "background-color:#fee2e2; color:#991b1b"
            if val == "Alkaline": return "background-color:#dbeafe; color:#1e40af"
            return ""

        st.dataframe(
            worst_display.style.applymap(highlight_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ All filtered stations are within the safe pH range (6.5–8.5).")


# # ---------------- Station Table (commented out) ---------------- #

# st.markdown("<h3 class='panel-heading'>Station Table</h3>", unsafe_allow_html=True)
# if not filtered.empty:
#     table = filtered[["location","ph","category","lat","lon"]].sort_values(by="ph", ascending=False)
#     st.dataframe(table, use_container_width=True, height=350)
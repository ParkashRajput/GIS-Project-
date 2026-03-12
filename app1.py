from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Water Quality Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BASEMAPS = {
    "Light": {"tiles": "CartoDB positron", "attr": "CartoDB"},
    "Street": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri",
    },
}

CATEGORY_ORDER = ["Acidic", "Safe", "Alkaline"]


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
    if ph < 6.5:
        return "Acidic"
    if ph <= 8.5:
        return "Safe"
    return "Alkaline"


def marker_color(ph):
    if ph < 6.5:
        return "#ef4444"
    if ph <= 8.5:
        return "#22c55e"
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

@st.cache_data(show_spinner=False)
def load_data():

    boundary = gpd.read_file(DATA_DIR / "hisar_new_boundary.geojson").to_crs("EPSG:4326")
    villages = gpd.read_file(DATA_DIR / "hisar_final_village_boundary.geojson").to_crs("EPSG:4326")
    stations = gpd.read_file(DATA_DIR / "hisar_points_updated.geojson")

    stations_wgs84 = stations.to_crs("EPSG:4326")
    stations_3857 = stations.to_crs("EPSG:3857")

    location_candidates = ["Location", "location", "Village", "village", "Name", "name"]
    location_col = next((c for c in location_candidates if c in stations_wgs84.columns), None)
    ph_col = next((c for c in stations_wgs84.columns if c.lower() == "ph"), None)

    stations_wgs84 = stations_wgs84.rename(
        columns={location_col: "location", ph_col: "ph"}
    ).copy()

    stations_wgs84["ph"] = stations_wgs84["ph"].astype(float)
    stations_wgs84 = stations_wgs84.dropna(subset=["geometry", "ph"])

    stations_wgs84["lat"] = stations_wgs84.geometry.y
    stations_wgs84["lon"] = stations_wgs84.geometry.x

    stations_wgs84["category"] = stations_wgs84["ph"].apply(classify_ph)
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


# ---------------- Map Builder ---------------- #

def build_map(boundary, villages, stations, center_lat, center_lon, basemap):

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
        ).add_to(m)

    legend_html = """
    <div class="map-legend">
        <div class="legend-title">pH Range</div>
        <div><span class="legend-dot acidic"></span>Acidic (&lt; 6.5)</div>
        <div><span class="legend-dot safe"></span>Safe (6.5 - 8.5)</div>
        <div><span class="legend-dot alkaline"></span>Alkaline (&gt; 8.5)</div>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

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


# ---------------- Load Everything ---------------- #

load_css()

boundary_gdf, village_gdf, station_gdf, station_gdf_3857 = load_data()


# ---------------- Hero Section ---------------- #

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

    station_list = sorted(station_gdf["location"].dropna().unique())
    
    search_station = st.selectbox(
        "Select Station",
        ["All Stations"] + station_list
    )


# ---------------- Filtering ---------------- #

filtered = station_gdf[
    (station_gdf["ph"] >= ph_range[0])
    & (station_gdf["ph"] <= ph_range[1])
    & (station_gdf["category"].isin(categories))
].copy()

if search_station != "All Stations":
    filtered = filtered[
        filtered["location"] == search_station
    ]

filtered_3857 = station_gdf_3857.loc[filtered.index]


# ---------------- Metrics ---------------- #

active_count = len(filtered)
safe_count = (filtered["category"] == "Safe").sum()
risk_count = (filtered["category"] != "Safe").sum()
avg_ph = filtered["ph"].mean() if not filtered.empty else 0

metric_cols = st.columns(4)

metric_cols[0].markdown(metric_block("Active Stations", active_count, "Filtered monitoring points"), unsafe_allow_html=True)
metric_cols[1].markdown(metric_block("Average pH", f"{avg_ph:.2f}", "District snapshot"), unsafe_allow_html=True)
metric_cols[2].markdown(metric_block("Safe Stations", safe_count, "Safe water range"), unsafe_allow_html=True)
metric_cols[3].markdown(metric_block("Attention Needed", risk_count, "Acidic or alkaline"), unsafe_allow_html=True)


# ---------------- Map ---------------- #

center_lat = filtered["lat"].mean() if not filtered.empty else station_gdf["lat"].mean()
center_lon = filtered["lon"].mean() if not filtered.empty else station_gdf["lon"].mean()

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
    )

    map_state = st_folium(
        m,
        height=700,
        use_container_width=True,
    )


with insight_col:

    st.markdown("<h3 class='panel-heading'>Station Insights</h3>", unsafe_allow_html=True)

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

        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">Focused Station</p>
                <h4>{selected_station['location']}</h4>
                <p><strong>pH:</strong> {selected_station['ph']:.2f}</p>
                <p><strong>Category:</strong> {selected_station['category']}</p>
                <p><strong>Latitude:</strong> {selected_station['lat']:.5f}</p>
                <p><strong>Longitude:</strong> {selected_station['lon']:.5f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(selected_station["ph"] / 14.0, 0), 1))


# ---------------- Chart ---------------- #

st.markdown("<h3 class='panel-heading'>Quality Distribution</h3>", unsafe_allow_html=True)

dist = filtered["category"].value_counts().reindex(
    CATEGORY_ORDER,
    fill_value=0,
)

st.bar_chart(dist)


# ---------------- Table ---------------- #

st.markdown("<h3 class='panel-heading'>Station Table</h3>", unsafe_allow_html=True)

if not filtered.empty:

    table = filtered[
        ["location", "ph", "category", "lat", "lon"]
    ].sort_values(
        by="ph",
        ascending=False,
    )

    st.dataframe(
        table,
        use_container_width=True,
        height=350,
    )
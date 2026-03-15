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


@st.cache_resource
def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


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


def metric_block(title, value, note):
    return (
        "<div class='metric-shell'>"
        f"<p class='metric-kicker'>{title}</p>"
        f"<p class='metric-number'>{value}</p>"
        f"<p class='metric-note'>{note}</p>"
        "</div>"
    )


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


def tooltip_html(row):
    return (
        "<div style='min-width:190px;padding:10px;border-radius:10px;"
        "background:#0f172a;color:#e2e8f0;border:1px solid #38bdf8;'>"
        f"<b>{row['location']}</b><br>"
        f"pH: {row['ph']:.2f}<br>"
        f"Category: {row['category']}<br>"
        f"Lat: {row['lat']:.4f}<br>"
        f"Lon: {row['lon']:.4f}"
        "</div>"
    )


def build_map(boundary, villages, stations, center_lat, center_lon, basemap):

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer(**BASEMAPS[basemap]).add_to(m)

    folium.GeoJson(
        boundary,
        name="District Boundary",
        style_function=lambda _: {
            "fillColor": "#67e8f9",
            "fillOpacity": 0.08,
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
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index:9999;
        background:white;
        padding:10px;
        border-radius:6px;
        font-size:14px;">
        <b>pH Legend</b><br>
        🔴 Acidic (<6.5)<br>
        🟢 Safe (6.5 - 8.5)<br>
        🔵 Alkaline (>8.5)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def nearest_station(stations_3857, stations_4326, lat, lon):

    if stations_3857.empty:
        return None

    click_point = gpd.GeoSeries(
        [Point(lon, lat)], crs="EPSG:4326"
    ).to_crs("EPSG:3857").iloc[0]

    idx = stations_3857.geometry.distance(click_point).idxmin()

    return stations_4326.loc[idx]


load_css()

boundary_gdf, village_gdf, station_gdf, station_gdf_3857 = load_data()

st.markdown(
"""
## Water Quality Intelligence Dashboard
Live monitoring of groundwater pH levels across Hisar district.
"""
)

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

    search_station = st.text_input("Search Station")

filtered = station_gdf[
    (station_gdf["ph"] >= ph_range[0])
    & (station_gdf["ph"] <= ph_range[1])
    & (station_gdf["category"].isin(categories))
].copy()

if search_station.strip():
    filtered = filtered[
        filtered["location"].str.contains(search_station, case=False, na=False)
    ]

filtered_3857 = station_gdf_3857.loc[filtered.index]

active_count = len(filtered)
safe_count = (filtered["category"] == "Safe").sum()
risk_count = (filtered["category"] != "Safe").sum()
avg_ph = filtered["ph"].mean() if not filtered.empty else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Active Stations", active_count)
c2.metric("Average pH", f"{avg_ph:.2f}")
c3.metric("Safe Stations", safe_count)
c4.metric("Attention Needed", risk_count)

center_lat = filtered["lat"].mean() if not filtered.empty else station_gdf["lat"].mean()
center_lon = filtered["lon"].mean() if not filtered.empty else station_gdf["lon"].mean()

map_col, info_col = st.columns([3, 1])

with map_col:

    st.subheader("Water Quality Map")

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
        height=650,
        use_container_width=True,
    )

with info_col:

    st.subheader("Station Insights")

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

        st.write("Location:", selected_station["location"])
        st.write("pH:", round(selected_station["ph"], 2))
        st.write("Category:", selected_station["category"])
        st.write("Latitude:", selected_station["lat"])
        st.write("Longitude:", selected_station["lon"])

st.subheader("Quality Distribution")

dist = filtered["category"].value_counts().reindex(
    CATEGORY_ORDER,
    fill_value=0,
)

st.bar_chart(dist)

st.subheader("Station Table")

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
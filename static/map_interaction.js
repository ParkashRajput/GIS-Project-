// Initialize map
var map = L.map('map').setView([30.37, 76.78], 10);

// Base map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: "© OpenStreetMap"
}).addTo(map);

// Color function based on pH
function getColor(ph){
    return ph < 6.5 ? "#ff0000" :
           ph < 7.5 ? "#00ff00" :
                      "#0066ff";
}

// Highlight region on hover
function highlightFeature(e){
    var layer = e.target;

    layer.setStyle({
        weight:3,
        color:"#000",
        fillOpacity:0.9
    });

    layer.openTooltip();
}

// Reset highlight
function resetHighlight(e){
    geojson.resetStyle(e.target);
}

// Click region popup
function regionClick(e){
    var layer = e.target;
    var ph = layer.feature.properties.ph;

    layer.bindPopup(`
        <div style="
            padding:12px;
            font-family:Arial;
            width:150px;
        ">
            <h3>Water Quality</h3>
            <b>pH Level:</b> ${ph}
        </div>
    `).openPopup();
}

// Attach events to region
function onEachFeature(feature, layer){

    layer.bindTooltip("pH: " + feature.properties.ph, {sticky:true});

    layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: regionClick
    });
}

// Example region GeoJSON
var geojson = L.geoJson(regionData, {

    style:function(feature){
        return{
            fillColor:getColor(feature.properties.ph),
            weight:1,
            color:"white",
            fillOpacity:0.6
        }
    },

    onEachFeature:onEachFeature

}).addTo(map);

// Add point markers
function addPoints(data){

    data.forEach(function(point){

        var marker = L.circleMarker(
            [point.lat, point.lon],
            {
                radius:6,
                color:"#000",
                fillColor:getColor(point.ph),
                fillOpacity:0.9
            }
        ).addTo(map);

        marker.bindTooltip("pH: " + point.ph);

        marker.on("mouseover",function(){
            this.setStyle({radius:10});
        });

        marker.on("mouseout",function(){
            this.setStyle({radius:6});
        });

        marker.on("click",function(){

            this.bindPopup(`
                <div style="
                    padding:12px;
                    font-family:Arial;
                    width:150px;
                ">
                    <h3>Water Sample</h3>
                    <b>pH Level:</b> ${point.ph}
                </div>
            `).openPopup();

        });

    });

}

// Example data
var points = [
    {lat:30.36, lon:76.77, ph:6.8},
    {lat:30.39, lon:76.80, ph:7.2},
    {lat:30.34, lon:76.75, ph:8.1}
];

// Add points to map
addPoints(points);
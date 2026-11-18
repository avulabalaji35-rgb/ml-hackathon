// PathSnap Evaluator Dashboard Application Logic (Laptop 2)

let mapInstance = null;
let currentTripId = null;

const apiStatusBadge = document.getElementById("api-status-badge");
const apiUrlInput = document.getElementById("api-url");
const tripSelect = document.getElementById("trip-select");
const btnRefresh = document.getElementById("btn-refresh");
const btnEvaluate = document.getElementById("btn-evaluate");

const metricAcc = document.getElementById("metric-acc");
const metricCont = document.getElementById("metric-cont");
const metricTime = document.getElementById("metric-time");
const metricArcs = document.getElementById("metric-arcs");

const infoTrip = document.getElementById("info-trip");
const infoFixes = document.getElementById("info-fixes");
const infoMsg = document.getElementById("info-msg");
const seqPreview = document.getElementById("sequence-preview");
const gtSeqPreview = document.getElementById("gt-sequence-preview");

// Initialize Leaflet Map
function initMap() {
  const mapContainer = document.getElementById("map-container");
  mapInstance = L.map(mapContainer).setView([50.0, 10.0], 6);

  // Esri World Street Map (works without API key)
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ',
    maxZoom: 19
  }).addTo(mapInstance);
}

// Check Backend Health
async function checkHealth() {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  apiStatusBadge.textContent = "Checking...";
  apiStatusBadge.className = "badge badge-checking";

  try {
    const res = await fetch(`${baseUrl}/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    apiStatusBadge.textContent = "API Online (Laptop 1)";
    apiStatusBadge.className = "badge badge-online";
    return true;
  } catch (err) {
    apiStatusBadge.textContent = "API Offline";
    apiStatusBadge.className = "badge badge-offline";
    console.error("Health check error:", err);
    return false;
  }
}

// Fetch Evaluation Trips List
async function fetchTrips() {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  tripSelect.innerHTML = '<option value="" disabled selected>Loading...</option>';

  const online = await checkHealth();
  if (!online) {
    tripSelect.innerHTML = '<option value="" disabled selected>Cannot connect to API</option>';
    return;
  }

  try {
    const res = await fetch(`${baseUrl}/api/trips`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    tripSelect.innerHTML = '';
    data.trips.forEach((tripId, index) => {
      const opt = document.createElement("option");
      opt.value = tripId;
      opt.textContent = `Trip ${tripId} (${index + 1} of ${data.trips.length})`;
      tripSelect.appendChild(opt);
    });

    if (data.trips.length > 0) {
      tripSelect.selectedIndex = 0;
      currentTripId = data.trips[0];
      infoTrip.textContent = currentTripId;
    }
  } catch (err) {
    tripSelect.innerHTML = '<option value="" disabled selected>Failed to fetch trips</option>';
    console.error("Trips fetch error:", err);
  }
}

// Run Evaluation for Selected Trip
async function runEvaluation() {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  const selectedTrip = tripSelect.value;
  if (!selectedTrip) {
    alert("Please select a trip first.");
    return;
  }

  btnEvaluate.disabled = true;
  btnEvaluate.textContent = "Processing via HMM...";
  infoMsg.textContent = `Sending inference request for Trip ${selectedTrip} to Laptop 1...`;

  try {
    const startTime = performance.now();
    const res = await fetch(`${baseUrl}/api/evaluate/${selectedTrip}`, {
      method: "POST"
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    const roundTripTime = Math.round(performance.now() - startTime);

    // Update KPI Metrics
    metricAcc.textContent = `${data.top1_accuracy_percent}%`;
    metricCont.textContent = `${data.route_continuity_percent}%`;
    metricTime.textContent = `${data.inference_time_ms} ms`;
    metricArcs.textContent = `${data.reconstructed_route_length} / ${data.ground_truth_route_length}`;

    infoTrip.textContent = data.trip_id;
    infoFixes.textContent = `${data.observations_count} points`;
    infoMsg.textContent = data.message;

    seqPreview.textContent = JSON.stringify(data.sample_predicted_arcs, null, 2);
    gtSeqPreview.textContent = JSON.stringify(data.sample_ground_truth_arcs, null, 2);

    // Render GPS Points on Map
    if (data.track_points && data.track_points.length > 0) {
      // Clear previous layers
      mapInstance.eachLayer(layer => {
        if (!layer._url) mapInstance.removeLayer(layer); // keep tile layer
      });

      const latlngs = data.track_points.map(pt => [pt.lat, pt.lon]);

      // Draw polyline
      const poly = L.polyline(latlngs, { color: '#0284c7', weight: 3, opacity: 0.8 }).addTo(mapInstance);

      // Add circle markers for first and last
      L.circleMarker(latlngs[0], { color: '#22c55e', radius: 7, fillOpacity: 0.9 })
        .bindPopup(`<b>Start</b><br>Trip ${data.trip_id}`)
        .addTo(mapInstance);

      L.circleMarker(latlngs[latlngs.length - 1], { color: '#ef4444', radius: 7, fillOpacity: 0.9 })
        .bindPopup(`<b>End</b><br>Trip ${data.trip_id}`)
        .addTo(mapInstance);

      mapInstance.fitBounds(poly.getBounds(), { padding: [30, 30] });
    }

    // Switch to complete interactive Folium visualization iframe
    const foliumFrame = document.getElementById("folium-frame");
    const mapContainer = document.getElementById("map-container");

    foliumFrame.src = `${baseUrl}/api/map/${selectedTrip}`;
    foliumFrame.classList.remove("hidden");
    mapContainer.classList.add("hidden");

  } catch (err) {
    alert(`Evaluation failed: ${err.message}`);
    infoMsg.textContent = `Error: ${err.message}`;
    console.error("Evaluation error:", err);
  } finally {
    btnEvaluate.disabled = false;
    btnEvaluate.textContent = "Run Map-Matching & Evaluation";
  }
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  fetchTrips();

  btnRefresh.addEventListener("click", fetchTrips);
  btnEvaluate.addEventListener("click", runEvaluation);
  tripSelect.addEventListener("change", (e) => {
    currentTripId = e.target.value;
    infoTrip.textContent = currentTripId;
  });
});

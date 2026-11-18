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


// ---------- anime.js motion layer ----------
const state = { acc: 0, cont: 0, time: 0 };
let badgePulse = null;

function countTo(key, value, render, el) {
  anime({
    targets: state,
    [key]: value,
    duration: 1100,
    easing: "easeOutExpo",
    update: () => { el.textContent = render(state[key]); }
  });
}

function growBar(id, pct) {
  anime({ targets: "#" + id, width: `${Math.min(100, pct)}%`, duration: 1100, easing: "easeOutExpo" });
}

function swapText(el, text) {
  anime({
    targets: el, opacity: [1, 0], translateY: [0, -4], duration: 120, easing: "easeInQuad",
    complete: () => {
      el.textContent = text;
      anime({ targets: el, opacity: [0, 1], translateY: [4, 0], duration: 260, easing: "easeOutQuad" });
    }
  });
}

function typeReveal(el, text) {
  const o = { n: 0 };
  anime({
    targets: o, n: text.length, duration: Math.min(900, 200 + text.length * 3), easing: "linear",
    update: () => { el.textContent = text.slice(0, Math.round(o.n)); }
  });
}

function pulseBadge(on) {
  if (badgePulse) { badgePulse.pause(); badgePulse = null; }
  anime.remove(apiStatusBadge);
  apiStatusBadge.style.opacity = 1;
  if (on) {
    badgePulse = anime({ targets: apiStatusBadge, opacity: [1, 0.4], duration: 700, direction: "alternate", loop: true, easing: "easeInOutSine" });
  } else {
    anime({ targets: apiStatusBadge, scale: [0.9, 1], duration: 500, easing: "easeOutElastic(1, .6)" });
  }
}

function introAnimation() {
  const tl = anime.timeline({ easing: "easeOutCubic" });
  tl.add({ targets: ".app-header", opacity: [0, 1], translateY: [-16, 0], duration: 500 })
    .add({ targets: ".logo-badge", scale: [0.4, 1], rotate: [-90, 0], duration: 700, easing: "easeOutElastic(1, .7)" }, "-=350")
    .add({ targets: ".toolbar-card", opacity: [0, 1], translateY: [16, 0], duration: 500 }, "-=450")
    .add({ targets: ".metric-card", opacity: [0, 1], translateY: [24, 0], scale: [0.96, 1], delay: anime.stagger(90), duration: 600 }, "-=300")
    .add({ targets: ".map-panel", opacity: [0, 1], translateX: [-24, 0], duration: 600 }, "-=350")
    .add({ targets: ".details-panel", opacity: [0, 1], translateX: [24, 0], duration: 600 }, "-=550")
    .add({ targets: ".legend-item", opacity: [0, 1], translateY: [6, 0], delay: anime.stagger(80), duration: 350 }, "-=300")
    .add({ targets: ".info-group", opacity: [0, 1], translateX: [10, 0], delay: anime.stagger(70), duration: 350 }, "-=500")
    .add({ targets: ".app-footer", opacity: [0, 1], duration: 400 }, "-=200");
}

// ---------- Toasts ----------
function toast(message, kind = "info", ttl = 3800) {
  const stack = document.getElementById("toast-stack");
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  el.style.opacity = 0;
  stack.appendChild(el);
  anime({ targets: el, opacity: [0, 1], translateX: [40, 0], duration: 380, easing: "easeOutBack" });
  setTimeout(() => {
    anime({
      targets: el, opacity: 0, translateX: 40, duration: 260, easing: "easeInQuad",
      complete: () => el.remove()
    });
  }, ttl);
}

// ---------- Theme toggle (persisted) ----------
function currentTheme() {
  const forced = document.documentElement.getAttribute("data-theme");
  if (forced) return forced;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  try {
    const saved = localStorage.getItem("pathsnap-theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage unavailable */ }

  const btn = document.getElementById("btn-theme");
  btn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    anime({
      targets: ".app-container", opacity: [1, 0.35], duration: 140, easing: "easeInQuad",
      complete: () => {
        document.documentElement.setAttribute("data-theme", next);
        try { localStorage.setItem("pathsnap-theme", next); } catch (e) { /* ignore */ }
        anime({ targets: ".app-container", opacity: [0.35, 1], duration: 320, easing: "easeOutQuad" });
      }
    });
    anime({ targets: btn, rotate: "+=180", duration: 450, easing: "easeOutBack" });
  });
}

function wireButtonMotion() {
  document.querySelectorAll(".btn").forEach(btn => {
    btn.addEventListener("mouseenter", () => anime({ targets: btn, scale: 1.04, translateY: -1, duration: 180, easing: "easeOutQuad" }));
    btn.addEventListener("mouseleave", () => anime({ targets: btn, scale: 1, translateY: 0, duration: 220, easing: "easeOutQuad" }));
    btn.addEventListener("mousedown", () => anime({ targets: btn, scale: 0.96, duration: 90, easing: "easeOutQuad" }));
    btn.addEventListener("mouseup", () => anime({ targets: btn, scale: 1.04, duration: 160, easing: "easeOutBack" }));
  });
  document.querySelectorAll(".metric-card").forEach(card => {
    card.addEventListener("mouseenter", () => anime({ targets: card, translateY: -4, duration: 220, easing: "easeOutQuad" }));
    card.addEventListener("mouseleave", () => anime({ targets: card, translateY: 0, duration: 260, easing: "easeOutQuad" }));
  });
  document.querySelectorAll(".input-field input, .input-field select").forEach(inp => {
    inp.addEventListener("focus", () => anime({ targets: inp, scale: 1.015, duration: 180, easing: "easeOutQuad" }));
    inp.addEventListener("blur", () => anime({ targets: inp, scale: 1, duration: 180, easing: "easeOutQuad" }));
  });
}

function showMapFrame(frame, container) {
  frame.classList.remove("hidden");
  container.classList.add("hidden");
  anime({ targets: frame, opacity: [0, 1], duration: 600, easing: "easeOutQuad" });
}

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
  pulseBadge(true);

  try {
    const res = await fetch(`${baseUrl}/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    apiStatusBadge.textContent = "API online";
    apiStatusBadge.className = "badge badge-online";
    pulseBadge(false);
    return true;
  } catch (err) {
    apiStatusBadge.textContent = "API offline";
    apiStatusBadge.className = "badge badge-offline";
    pulseBadge(false);
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
    toast("Please select a trip first.", "error");
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
    countTo("acc", data.top1_accuracy_percent, v => `${v.toFixed(2)}%`, metricAcc);
    countTo("cont", data.route_continuity_percent, v => `${v.toFixed(2)}%`, metricCont);
    countTo("time", data.inference_time_ms, v => `${Math.round(v)} ms`, metricTime);
    const arcsState = { rec: 0, gt: 0 };
    anime({
      targets: arcsState, rec: data.reconstructed_route_length, gt: data.ground_truth_route_length,
      duration: 1100, easing: "easeOutExpo",
      update: () => { metricArcs.textContent = `${Math.round(arcsState.rec)} / ${Math.round(arcsState.gt)}`; }
    });
    growBar("bar-acc", data.top1_accuracy_percent);
    growBar("bar-cont", data.route_continuity_percent);

    swapText(infoTrip, String(data.trip_id));
    swapText(infoFixes, `${data.observations_count} points`);
    swapText(infoMsg, data.message);

    typeReveal(seqPreview, JSON.stringify(data.sample_predicted_arcs, null, 2));
    typeReveal(gtSeqPreview, JSON.stringify(data.sample_ground_truth_arcs, null, 2));

    // Render GPS Points on Map
    if (data.track_points && data.track_points.length > 0) {
      // Clear previous layers
      mapInstance.eachLayer(layer => {
        if (!layer._url) mapInstance.removeLayer(layer); // keep tile layer
      });

      const latlngs = data.track_points.map(pt => [pt.lat, pt.lon]);

      // Draw polyline
      const poly = L.polyline(latlngs, { color: '#2563eb', weight: 3, opacity: 0.8 }).addTo(mapInstance);

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
    showMapFrame(foliumFrame, mapContainer);
    toast(`Trip ${data.trip_id} evaluated in ${data.inference_time_ms} ms`, "success");

  } catch (err) {
    toast(`Evaluation failed: ${err.message}`, "error", 6000);
    infoMsg.textContent = `Error: ${err.message}`;
    console.error("Evaluation error:", err);
  } finally {
    btnEvaluate.disabled = false;
    btnEvaluate.textContent = "Run evaluation";
  }
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  // If loaded via HTTP (e.g. from Laptop 1's server), automatically use the current origin
  if (window.location.protocol.startsWith("http")) {
    apiUrlInput.value = window.location.origin;
  }
  initMap();
  introAnimation();
  initTheme();
  wireButtonMotion();
  fetchTrips();

  // Keyboard shortcut: Ctrl/Cmd+Enter runs the evaluation
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (!btnEvaluate.disabled) runEvaluation();
    }
  });
  btnEvaluate.title = "Ctrl+Enter";

  btnRefresh.addEventListener("click", fetchTrips);
  btnEvaluate.addEventListener("click", runEvaluation);
  tripSelect.addEventListener("change", (e) => {
    currentTripId = e.target.value;
    infoTrip.textContent = currentTripId;
  });
});

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.message || "Request failed");
  return data;
}

function renderRows(rows) {
  if (!rows.length) return "<p>No records yet.</p>";
  const keys = Object.keys(rows[0]);
  return `<table><tr>${keys.map(k => `<th>${k}</th>`).join("")}</tr>` +
    rows.map(row => `<tr>${keys.map(k => `<td>${row[k] ?? ""}</td>`).join("")}</tr>`).join("") +
    "</table>";
}

async function refresh() {
  try {
    const status = await getJson("/api/status");
    const statusNode = document.querySelector("#status");
    if (statusNode) statusNode.textContent = `Backend online · ${status.cnn_model ? "CNN ready" : "CNN unavailable"}`;
    const dashboard = await getJson("/api/dashboard");
    const values = {
      active: dashboard.active_vehicles, vehicles: dashboard.total_vehicles,
      "available-spaces": dashboard.available_spaces,
      entries: dashboard.todays_entries, exits: dashboard.todays_exits,
    };
    Object.entries(values).forEach(([id, value]) => {
      const node = document.querySelector(`#${id}`);
      if (node) node.textContent = value;
    });
    const utilization = document.querySelector("#dashboard-utilization");
    if (utilization) utilization.textContent =
      `Parking utilization: ${dashboard.parking_utilization_percent}%`;
    const recent = document.querySelector("#recent");
    if (recent) recent.innerHTML = renderRows((await getJson("/api/parking/sessions")).slice(0, 5));
    const analytics = document.querySelector("#analytics");
    if (analytics) analytics.innerHTML = `<pre>${JSON.stringify(dashboard, null, 2)}</pre>`;
    ["entries", "exits", "active", "vehicles"].forEach((key) => {
      const node = document.querySelector(`#analytics-${key}`);
      if (node) node.textContent = key === "entries" ? dashboard.todays_entries :
        key === "exits" ? dashboard.todays_exits :
        key === "active" ? dashboard.active_vehicles : dashboard.total_vehicles;
    });
  } catch (error) {
    const status = document.querySelector("#status");
    if (status) status.textContent = "Backend error";
  }
}

async function setMode(mode) {
  const result = await getJson("/api/parking/mode", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({mode})
  });
  const target = document.querySelector("#mode");
  if (target) target.textContent = ` Current mode: ${result.mode}`;
}

async function loadTable() {
  const table = document.querySelector("#table");
  if (!table) return;
  const search = document.querySelector("#table-search")?.value || "";
  const params = new URLSearchParams({search});
  const date = document.querySelector("#history-date")?.value;
  const status = document.querySelector("#history-status")?.value;
  if (date) params.set("date", date);
  if (status) params.set("status", status);
  table.innerHTML = renderRows(await getJson(`${table.dataset.endpoint}?${params}`));
}

function renderRecognition(data) {
  const status = document.querySelector("#recognition-status");
  const plate = document.querySelector("#recognition-plate");
  const confidence = document.querySelector("#recognition-confidence");
  const chars = document.querySelector("#recognition-characters");
  if (status) status.textContent = data.message || data.status;
  if (plate) plate.textContent = data.plate_number || "—";
  if (confidence) confidence.textContent = data.confidence ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : "";
  if (chars) chars.innerHTML = (data.characters || []).map(item =>
    `<span class="character-chip">${item.character} ${(item.confidence * 100).toFixed(0)}%</span>`).join("");
  for (const [id, source] of [["detected-plate", data.plate_image], ["segmented-characters", data.characters_image]]) {
    const image = document.querySelector(`#${id}`);
    if (image) { image.src = source || ""; image.hidden = !source; }
  }
  const process = document.querySelector("#process-recognition");
  if (process) process.hidden =
    !data.plate_number || data.status !== "recognized" || data.processing_status === "processed";
  if (process) process.dataset.plate = data.plate_number || "";
  const debug = document.querySelector("#recognition-debug");
  const debugImages = document.querySelector("#debug-images");
  const debugPrediction = document.querySelector("#debug-prediction");
  const entries = Object.entries(data.debug_images || {});
  if (debug && debugImages) {
    debug.hidden = !entries.length;
    if (debugPrediction) debugPrediction.textContent =
      `Prediction: ${data.plate_number || "none"} · Confidence: ${((data.confidence || 0) * 100).toFixed(1)}% · Status: ${data.status}`;
    debugImages.innerHTML = entries.map(([name, source]) =>
      `<figure><img class="result-image" src="${source}" alt="${name}"><figcaption>${name.replaceAll("_", " ")}</figcaption></figure>`
    ).join("");
  }
}

async function loadCameras() {
  const select = document.querySelector("#camera-select");
  if (!select) return;
  const cameras = await getJson("/api/cameras");
  select.innerHTML = cameras.length ? cameras.map(c => `<option value="${c.id}">${c.name} · index ${c.index} · ${c.backend} ${c.frame_size ? `(${c.frame_size.width}x${c.frame_size.height})` : ""}</option>`).join("") :
    '<option value="">No cameras detected</option>';
}

async function setupCameraPage() {
  const select = document.querySelector("#camera-select");
  if (!select) return;
  await loadCameras();
  const message = text => { document.querySelector("#camera-message").textContent = text; };
  const cameraStatus = document.querySelector("#camera-status");
  const updateCameraStatus = data => {
    if (cameraStatus) cameraStatus.textContent = data.connected
      ? `Connected: ${data.name || `Camera ${data.index}`} · ${data.backend || "OpenCV"}`
      : `Not connected${data.error ? ` · ${data.error}` : ""}`;
  };
  document.querySelector("#camera-refresh").onclick = async () => {
    try { await loadCameras(); message("Camera list refreshed."); }
    catch (error) { message(`Camera scan failed: ${error.message}`); }
  };
  const pollCameraStatus = async () => {
    try { updateCameraStatus(await getJson("/api/cameras/status")); }
    catch (error) { if (cameraStatus) cameraStatus.textContent = `Camera status unavailable: ${error.message}`; }
  };
  const pollRecognition = async () => {
    try {
      const result = await getJson("/api/camera/recognition");
      if (result.latest_detection) renderRecognition(result.latest_detection);
      if (result.latest_detection?.processing_message) message(result.latest_detection.processing_message);
    } catch (error) { message(`Recognition status unavailable: ${error.message}`); }
  };
  document.querySelector("#mode-entry").onclick = () => setMode("entry");
  document.querySelector("#mode-exit").onclick = () => setMode("exit");
  document.querySelector("#camera-start").onclick = async () => {
    try {
      const result = await getJson("/api/cameras/connect", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({device_id: select.value})});
      updateCameraStatus(result);
      message("Camera connected and returning frames.");
    }
    catch (error) { message(error.message); }
  };
  document.querySelector("#camera-stop").onclick = async () => {
    const result = await getJson("/api/cameras/disconnect", {method: "POST"});
    updateCameraStatus(result);
    message("Camera disconnected.");
  };
  document.querySelector("#camera-capture").onclick = async () => {
    try { await getJson("/api/camera/capture", {method: "POST"}); message("Frame captured. Click Recognize plate."); }
    catch (error) { message(error.message); }
  };
  document.querySelector("#camera-recognize").onclick = async () => {
    try {
      const debug = document.querySelector("#camera-debug")?.checked ? "?debug=1" : "";
      renderRecognition(await getJson(`/api/camera/recognize${debug}`, {method: "POST"}));
    }
    catch (error) { renderRecognition({status: "failed", message: error.message}); }
  };
  const process = document.querySelector("#process-recognition");
  if (process) process.onclick = async () => {
    try {
      const endpoint = (await getJson("/api/parking/mode")).mode === "entry" ? "/api/parking/entry" : "/api/parking/exit";
      const result = await getJson(endpoint, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({plate: process.dataset.plate})});
      message(`${endpoint.endsWith("entry") ? "Entry" : "Exit"} processed successfully.`);
      renderRecognition({...result, status: "processed", message: "Parking operation completed"});
    } catch (error) { message(error.message); }
  };
  pollCameraStatus();
  setInterval(pollCameraStatus, 2000);
  pollRecognition();
  setInterval(pollRecognition, 1000);
}

async function setupSettings() {
  const form = document.querySelector("#settings-form");
  if (!form) return;
  const settings = await getJson("/api/settings");
  for (const [key, value] of Object.entries(settings)) if (form.elements[key]) form.elements[key].value = value;
  form.onsubmit = async event => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(form).entries());
    try { await getJson("/api/settings", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)}); document.querySelector("#settings-message").textContent = "Settings saved."; }
    catch (error) { document.querySelector("#settings-message").textContent = error.message; }
  };
  const cameraList = document.querySelector("#camera-list");
  if (cameraList) {
    try {
      const cameras = await getJson("/api/cameras");
      cameraList.innerHTML = cameras.length
        ? renderRows(cameras)
        : "<p>No available cameras returned.</p>";
    } catch (error) { cameraList.textContent = error.message; }
  }
}

refresh();
loadTable();
setupCameraPage();
setupSettings();
setInterval(refresh, 5000);

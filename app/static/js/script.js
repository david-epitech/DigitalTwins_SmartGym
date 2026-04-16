const state = {
  profile: "Intermediate",
  exercise: "Squat",
  heartRate: 135,
  fatigue: 55,
  load: 80,
  reps: 8,
  recoveryScore: 70,
  hydrationLevel: 75,
  sleepQuality: 70,
  sessionIntensity: 60,
  powerOutput: 520,
  injuryRisk: 0,
};

const metricConfig = [
  { key: "heartRate", label: "Heart Rate", unit: "bpm" },
  { key: "fatigue", label: "Fatigue", unit: "%" },
  { key: "load", label: "Load", unit: "kg" },
  { key: "reps", label: "Reps", unit: "" },
  { key: "recoveryScore", label: "Recovery Score", unit: "%" },
  { key: "hydrationLevel", label: "Hydration Level", unit: "%" },
  { key: "powerOutput", label: "Power Output", unit: "W" },
];

const ui = {
  tabButtons: document.querySelectorAll(".tab-btn"),
  tabs: document.querySelectorAll(".tab-page"),
  metricCards: document.getElementById("metricCards"),
  riskValue: document.getElementById("riskValue"),
  riskLevel: document.getElementById("riskLevel"),
  predictedState: document.getElementById("predictedState"),
  recommendation: document.getElementById("recommendation"),
  coachInsight: document.getElementById("coachInsight"),
  modelConnected: document.getElementById("modelConnected"),
  globalStatus: document.getElementById("globalStatus"),
  selectedAthleteTop: document.getElementById("selectedAthleteTop"),
  featureList: document.getElementById("featureList"),
  modelSummaryList: document.getElementById("modelSummaryList"),
  metricsCards: document.getElementById("metricsCards"),
  viewerFallback: document.getElementById("viewerFallback"),
  controls: {
    athleteProfile: document.getElementById("athleteProfile"),
    exerciseType: document.getElementById("exerciseType"),
    sessionIntensity: document.getElementById("sessionIntensity"),
    recoveryScore: document.getElementById("recoveryScore"),
    hydrationLevel: document.getElementById("hydrationLevel"),
    sleepQuality: document.getElementById("sleepQuality"),
    heartRate: document.getElementById("heartRate"),
    fatigue: document.getElementById("fatigue"),
    load: document.getElementById("load"),
    reps: document.getElementById("reps"),
    powerOutput: document.getElementById("powerOutput"),
    intensityLabel: document.getElementById("intensityLabel"),
    recoveryLabel: document.getElementById("recoveryLabel"),
    hydrationLabel: document.getElementById("hydrationLabel"),
    sleepLabel: document.getElementById("sleepLabel"),
    predictBtn: document.getElementById("predictBtn"),
    simulateBtn: document.getElementById("simulateBtn"),
    resetBtn: document.getElementById("resetBtn"),
  },
};

let scene;
let camera;
let renderer;
let controls;
let athleteModel;

let heartRateChart;
let fatigueChart;
let riskChart;

function valueClass(value, key) {
  if (["fatigue", "heartRate"].includes(key)) {
    if (value > 75 || (key === "heartRate" && value > 165)) return "danger";
    if (value > 55 || (key === "heartRate" && value > 140)) return "warn";
    return "good";
  }

  if (["recoveryScore", "hydrationLevel"].includes(key)) {
    if (value < 40) return "danger";
    if (value < 65) return "warn";
    return "good";
  }

  if (key === "powerOutput") {
    if (value > 900) return "danger";
    if (value > 650) return "warn";
    return "good";
  }

  return "good";
}

function renderMetricCards() {
  ui.metricCards.innerHTML = "";

  metricConfig.forEach((metric) => {
    const raw = state[metric.key];
    const value = typeof raw === "number" ? raw.toFixed(metric.key === "reps" ? 0 : 0) : raw;

    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-title">${metric.label}</div>
      <div class="metric-value ${valueClass(raw, metric.key)}">${value}${metric.unit ? ` ${metric.unit}` : ""}</div>
    `;
    ui.metricCards.appendChild(card);
  });
}

function updateRiskUI(prediction) {
  state.injuryRisk = prediction.injury_risk;
  ui.riskValue.textContent = `${prediction.injury_risk.toFixed(1)}%`;
  ui.riskLevel.textContent = prediction.risk_level;
  ui.predictedState.textContent = prediction.predicted_state;
  ui.recommendation.textContent = prediction.recommendation;
  ui.coachInsight.textContent = prediction.coach_insight;

  ui.riskLevel.className = `risk-badge ${prediction.risk_level.toLowerCase()}`;

  if (prediction.risk_level === "HIGH") {
    ui.globalStatus.className = "status-pill status-alert";
    ui.globalStatus.textContent = "ALERT";
  } else if (prediction.risk_level === "MEDIUM") {
    ui.globalStatus.className = "status-pill status-monitoring";
    ui.globalStatus.textContent = "MONITORING";
  } else {
    ui.globalStatus.className = "status-pill status-active";
    ui.globalStatus.textContent = "ACTIVE";
  }
}

function payloadFromControls() {
  return {
    athlete_profile: ui.controls.athleteProfile.value,
    exercise_type: ui.controls.exerciseType.value,
    heart_rate: Number(ui.controls.heartRate.value),
    fatigue: Number(ui.controls.fatigue.value),
    load: Number(ui.controls.load.value),
    reps: Number(ui.controls.reps.value),
    recovery_score: Number(ui.controls.recoveryScore.value),
    hydration_level: Number(ui.controls.hydrationLevel.value),
    sleep_quality: Number(ui.controls.sleepQuality.value),
    session_intensity: Number(ui.controls.sessionIntensity.value),
    power_output: Number(ui.controls.powerOutput.value),
  };
}

function syncStateFromControls() {
  state.profile = ui.controls.athleteProfile.value;
  state.exercise = ui.controls.exerciseType.value;
  state.heartRate = Number(ui.controls.heartRate.value);
  state.fatigue = Number(ui.controls.fatigue.value);
  state.load = Number(ui.controls.load.value);
  state.reps = Number(ui.controls.reps.value);
  state.recoveryScore = Number(ui.controls.recoveryScore.value);
  state.hydrationLevel = Number(ui.controls.hydrationLevel.value);
  state.sleepQuality = Number(ui.controls.sleepQuality.value);
  state.sessionIntensity = Number(ui.controls.sessionIntensity.value);
  state.powerOutput = Number(ui.controls.powerOutput.value);
  ui.selectedAthleteTop.textContent = state.profile;
}

function setupTabs() {
  ui.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      ui.tabButtons.forEach((btn) => btn.classList.remove("active"));
      ui.tabs.forEach((tab) => tab.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.tab).classList.add("active");
    });
  });
}

function setupControls() {
  const bindNumberControl = (field) => {
    field.addEventListener("input", () => {
      syncStateFromControls();
      renderMetricCards();
    });
  };

  [
    ui.controls.athleteProfile,
    ui.controls.exerciseType,
    ui.controls.heartRate,
    ui.controls.fatigue,
    ui.controls.load,
    ui.controls.reps,
    ui.controls.powerOutput,
  ].forEach(bindNumberControl);

  const bindSlider = (slider, label) => {
    slider.addEventListener("input", () => {
      label.textContent = slider.value;
      syncStateFromControls();
      renderMetricCards();
    });
  };

  bindSlider(ui.controls.sessionIntensity, ui.controls.intensityLabel);
  bindSlider(ui.controls.recoveryScore, ui.controls.recoveryLabel);
  bindSlider(ui.controls.hydrationLevel, ui.controls.hydrationLabel);
  bindSlider(ui.controls.sleepQuality, ui.controls.sleepLabel);

  ui.controls.simulateBtn.addEventListener("click", () => {
    ui.controls.sessionIntensity.value = 90;
    ui.controls.recoveryScore.value = 30;
    ui.controls.hydrationLevel.value = 35;
    ui.controls.sleepQuality.value = 40;
    ui.controls.heartRate.value = 176;
    ui.controls.fatigue.value = 88;
    ui.controls.load.value = 150;
    ui.controls.reps.value = 12;
    ui.controls.powerOutput.value = 980;

    ui.controls.intensityLabel.textContent = "90";
    ui.controls.recoveryLabel.textContent = "30";
    ui.controls.hydrationLabel.textContent = "35";
    ui.controls.sleepLabel.textContent = "40";

    syncStateFromControls();
    renderMetricCards();
  });

  ui.controls.resetBtn.addEventListener("click", () => {
    ui.controls.athleteProfile.value = "Intermediate";
    ui.controls.exerciseType.value = "Squat";
    ui.controls.sessionIntensity.value = 60;
    ui.controls.recoveryScore.value = 70;
    ui.controls.hydrationLevel.value = 75;
    ui.controls.sleepQuality.value = 70;
    ui.controls.heartRate.value = 135;
    ui.controls.fatigue.value = 55;
    ui.controls.load.value = 80;
    ui.controls.reps.value = 8;
    ui.controls.powerOutput.value = 520;

    ui.controls.intensityLabel.textContent = "60";
    ui.controls.recoveryLabel.textContent = "70";
    ui.controls.hydrationLabel.textContent = "75";
    ui.controls.sleepLabel.textContent = "70";

    syncStateFromControls();
    renderMetricCards();
  });

  ui.controls.predictBtn.addEventListener("click", predictRisk);
}

function initCharts() {
  const labels = ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t0"];

  const configBase = (label, data, color) => ({
    type: "line",
    data: {
      labels,
      datasets: [{
        label,
        data,
        borderColor: color,
        backgroundColor: `${color}33`,
        fill: true,
        tension: 0.35,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2.4,
      plugins: { legend: { labels: { color: "#d9e2f2" } } },
      scales: {
        x: { ticks: { color: "#9fb0cc" }, grid: { color: "#1e2c4e" } },
        y: { ticks: { color: "#9fb0cc" }, grid: { color: "#1e2c4e" } },
      },
    },
  });

  heartRateChart = new Chart(document.getElementById("heartRateChart"), configBase("Heart Rate", Array(10).fill(state.heartRate), "#4f8cff"));
  fatigueChart = new Chart(document.getElementById("fatigueChart"), configBase("Fatigue", Array(10).fill(state.fatigue), "#f5a524"));
  riskChart = new Chart(document.getElementById("riskChart"), configBase("Injury Risk", Array(10).fill(0), "#ef4444"));
}

function pushChartValue(chart, value) {
  const dataset = chart.data.datasets[0].data;
  dataset.shift();
  dataset.push(value);
  chart.update();
}

function updateChartsAfterPrediction(risk) {
  pushChartValue(heartRateChart, state.heartRate);
  pushChartValue(fatigueChart, state.fatigue);
  pushChartValue(riskChart, risk);
}

async function predictRisk() {
  syncStateFromControls();
  renderMetricCards();

  ui.controls.predictBtn.disabled = true;
  ui.controls.predictBtn.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromControls()),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    updateRiskUI(data);
    updateChartsAfterPrediction(data.injury_risk);
  } catch (error) {
    ui.modelConnected.textContent = `Prediction error: ${error.message}`;
    ui.modelConnected.style.background = "rgba(239, 68, 68, 0.25)";
    ui.modelConnected.style.color = "#ffb3b3";
  } finally {
    ui.controls.predictBtn.disabled = false;
    ui.controls.predictBtn.textContent = "Predict Risk";
  }
}

function init3DViewer() {
  if (!window.THREE || !window.THREE.OrbitControls || !window.THREE.FBXLoader || !window.fflate) {
    ui.viewerFallback.classList.remove("hidden");
    ui.viewerFallback.textContent = "3D libraries failed to load.";
    return;
  }

  const container = document.getElementById("viewer3d");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a1124);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 120, 270);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
  scene.add(ambientLight);

  const directional = new THREE.DirectionalLight(0xffffff, 1);
  directional.position.set(120, 170, 80);
  scene.add(directional);

  const fill = new THREE.DirectionalLight(0x8fa8ff, 0.5);
  fill.position.set(-80, 60, -120);
  scene.add(fill);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 70, 0);

  const loader = new THREE.FBXLoader();
  loader.load(
    "/static/model/gym.fbx",
    (fbx) => {
      athleteModel = fbx;
      athleteModel.scale.setScalar(0.45);
      athleteModel.position.set(0, -20, 0);
      scene.add(athleteModel);
    },
    undefined,
    () => {
      ui.viewerFallback.classList.remove("hidden");
    }
  );

  window.addEventListener("resize", onViewerResize);
  animateViewer();
}

function onViewerResize() {
  const container = document.getElementById("viewer3d");
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animateViewer() {
  requestAnimationFrame(animateViewer);
  if (athleteModel) {
    athleteModel.rotation.y += 0.0025;
  }
  controls.update();
  renderer.render(scene, camera);
}

function fillModelTab(payload) {
  ui.modelSummaryList.innerHTML = `
    <li><strong>Model type:</strong> ${payload.model_type}</li>
    <li><strong>Task:</strong> ${payload.task}</li>
    <li><strong>Target:</strong> ${payload.target}</li>
    <li><strong>Input type:</strong> ${payload.input_type}</li>
    <li><strong>Output:</strong> ${payload.output}</li>
  `;

  ui.featureList.innerHTML = "";
  payload.features.forEach((feature) => {
    const item = document.createElement("li");
    item.textContent = feature;
    ui.featureList.appendChild(item);
  });

  ui.metricsCards.innerHTML = "";
  Object.entries(payload.metrics).forEach(([name, value]) => {
    const card = document.createElement("div");
    card.className = "mini-card";
    card.innerHTML = `<span>${name}</span><strong>${Number(value).toFixed(4)}</strong>`;
    ui.metricsCards.appendChild(card);
  });
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    const payload = await response.json();
    fillModelTab(payload);

    if (!payload.model_available) {
      ui.modelConnected.textContent = "LSTM model not found. Run python main.py first.";
      ui.modelConnected.style.background = "rgba(245, 165, 36, 0.25)";
      ui.modelConnected.style.color = "#ffd59a";
    }
  } catch (error) {
    ui.modelConnected.textContent = "Cannot load model info";
    ui.modelConnected.style.background = "rgba(239, 68, 68, 0.25)";
    ui.modelConnected.style.color = "#ffb3b3";
  }
}

function init() {
  setupTabs();
  setupControls();
  syncStateFromControls();
  renderMetricCards();
  initCharts();
  init3DViewer();
  loadModelInfo();
}

init();

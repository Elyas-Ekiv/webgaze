const fileInput = document.querySelector("#file");
const urlInput = document.querySelector("#url");
const predictButton = document.querySelector("#predict");
const sampleButton = document.querySelector("#sample");
const status = document.querySelector("#status");
const canvas = document.querySelector("#view");
const opacity = document.querySelector("#opacity");
const modelName = document.querySelector("#model-name");
const regions = document.querySelector("#regions");
const boxesToggle = document.querySelector("#boxes");
const ctx = canvas.getContext("2d");

const TYPE_COLOR = {
  header: "#0e5c5a",
  text_or_content: "#1d4e89",
  layout_region: "#c47a12",
  background: "#8a8175",
};

let screenshot = null;
let heatmap = null;
let regionBoxes = [];

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

const PLACE_ROWS = ["top", "middle", "bottom"];
const PLACE_COLS = ["left", "center", "right"];

function placeIndex(region) {
  const cx = region.x + region.width / 2;
  const cy = region.y + region.height / 2;
  const col = cx < 128 ? 0 : cx > 256 ? 2 : 1;
  const row = cy < 72 ? 0 : cy > 144 ? 2 : 1;
  return row * 3 + col;
}

function draw() {
  if (!screenshot) return;
  canvas.width = screenshot.width;
  canvas.height = screenshot.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(screenshot, 0, 0);
  if (heatmap) {
    ctx.globalAlpha = Number(opacity.value);
    ctx.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
  }
  if (!boxesToggle.checked) return;
  const scaleX = canvas.width / 384;
  const scaleY = canvas.height / 216;
  ctx.lineWidth = 2;
  regionBoxes.forEach((region) => {
    const color = TYPE_COLOR[region.element_type] || TYPE_COLOR.layout_region;
    ctx.strokeStyle = color;
    ctx.strokeRect(region.x * scaleX, region.y * scaleY, region.width * scaleX, region.height * scaleY);
  });
}

async function show(payload) {
  screenshot = await loadImage(payload.screenshot);
  heatmap = await loadImage(payload.heatmap);
  modelName.textContent = payload.model;
  regionBoxes = payload.regions;
  regions.innerHTML = "";
  const chipSlots = PLACE_ROWS.flatMap((row) => PLACE_COLS.map((col) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    const title = document.createElement("strong");
    title.textContent = `${row} ${col}`;
    const chips = document.createElement("div");
    chips.className = "chips";
    cell.append(title, chips);
    regions.appendChild(cell);
    return chips;
  }));
  payload.regions.slice(0, 18).forEach((region) => {
    const chip = document.createElement("span");
    chip.dataset.type = region.element_type;
    chip.textContent = region.element_type.replaceAll("_", " ");
    const slot = chipSlots[placeIndex(region)];
    slot.appendChild(chip);
    slot.parentElement.classList.add("filled");
  });
  draw();
}

async function submit(formData, endpoint) {
  predictButton.disabled = true;
  sampleButton.disabled = true;
    status.textContent = "Running…";
  try {
    const response = await fetch(endpoint, { method: formData ? "POST" : "GET", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Prediction failed");
    await show(payload);
    status.textContent = "Done — use the slider to blend the heatmap.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    sampleButton.disabled = false;
  }
}

predictButton.addEventListener("click", () => {
  const data = new FormData();
  if (urlInput.value.trim()) data.append("url", urlInput.value.trim());
  if (fileInput.files[0]) data.append("file", fileInput.files[0]);
  if (!urlInput.value.trim() && !fileInput.files[0]) {
    status.textContent = "Choose an image or enter a URL.";
    return;
  }
  submit(data, "/api/predict");
});

sampleButton.addEventListener("click", () => submit(null, "/api/sample"));
opacity.addEventListener("input", draw);
boxesToggle.addEventListener("change", draw);

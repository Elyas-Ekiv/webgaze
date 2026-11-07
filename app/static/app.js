(() => {
  const TYPE_COLORS = {
    header: "#0f6b5c",
    text_or_content: "#1a5fbf",
    layout_region: "#b86b00",
    background: "#6a756f",
  };

  const TYPE_LABELS = {
    header: "Header",
    text_or_content: "Text / content",
    layout_region: "Layout",
    background: "Background",
  };

  const els = {
    modeTabs: [...document.querySelectorAll(".mode-tab")],
    modeScreenshot: document.getElementById("mode-screenshot"),
    modeUrl: document.getElementById("mode-url"),
    dropzone: document.getElementById("dropzone"),
    file: document.getElementById("file"),
    previewWrap: document.getElementById("preview-wrap"),
    preview: document.getElementById("preview"),
    clearPreview: document.getElementById("clear-preview"),
    sampleBtn: document.getElementById("sample-btn"),
    url: document.getElementById("url"),
    predictBtn: document.getElementById("predict-btn"),
    status: document.getElementById("status"),
    emptyState: document.getElementById("empty-state"),
    results: document.getElementById("results"),
    modelBadge: document.getElementById("model-badge"),
    sourceBadge: document.getElementById("source-badge"),
    opacity: document.getElementById("opacity"),
    showBoxes: document.getElementById("show-boxes"),
    showHotspots: document.getElementById("show-hotspots"),
    hotspots: document.getElementById("hotspots"),
    structureCounts: document.getElementById("structure-counts"),
    domToggle: document.getElementById("dom-toggle"),
    domPanel: document.getElementById("dom-panel"),
    domHtml: document.getElementById("dom-html"),
    dlOverlay: document.getElementById("dl-overlay"),
    dlHeatmap: document.getElementById("dl-heatmap"),
    footerMetrics: document.getElementById("footer-metrics"),
    canvasOriginal: document.getElementById("canvas-original"),
    canvasHeatmap: document.getElementById("canvas-heatmap"),
    canvasOverlay: document.getElementById("canvas-overlay"),
    stepInput: document.getElementById("step-input"),
    stepRun: document.getElementById("step-run"),
    stepOutput: document.getElementById("step-output"),
  };

  const state = {
    mode: "screenshot",
    file: null,
    previewUrl: null,
    payload: null,
    images: { screenshot: null, heatmap: null },
    activeHotspot: -1,
    busy: false,
    phase: "input",
  };

  function setWorkflow(phase) {
    state.phase = phase;
    const map = {
      input: els.stepInput,
      run: els.stepRun,
      output: els.stepOutput,
    };
    [els.stepInput, els.stepRun, els.stepOutput].forEach((el) => {
      el.classList.remove("active", "done");
    });
    if (phase === "input") {
      els.stepInput.classList.add("active");
      els.stepRun.disabled = true;
      els.stepOutput.disabled = true;
    } else if (phase === "run") {
      els.stepInput.classList.add("done");
      els.stepRun.classList.add("active");
      els.stepRun.disabled = false;
      els.stepOutput.disabled = true;
    } else {
      els.stepInput.classList.add("done");
      els.stepRun.classList.add("done");
      els.stepOutput.classList.add("active");
      els.stepRun.disabled = false;
      els.stepOutput.disabled = false;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    els.modeTabs.forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    els.modeScreenshot.classList.toggle("hidden", mode !== "screenshot");
    els.modeUrl.classList.toggle("hidden", mode !== "url");
  }

  function setBusy(busy) {
    state.busy = busy;
    els.predictBtn.disabled = busy;
    els.sampleBtn.disabled = busy;
  }

  function setStatusIdle(message) {
    els.status.innerHTML = `<p class="status-idle">${message}</p>`;
  }

  function setStatusError(message) {
    els.status.innerHTML = `<p class="status-error">${message}</p>`;
    setWorkflow("input");
  }

  function setStatusSteps(steps, activeIndex) {
    const items = steps
      .map((label, i) => {
        let cls = "";
        if (i < activeIndex) cls = "done";
        else if (i === activeIndex) cls = "active";
        return `<li class="${cls}"><span class="dot"></span><span>${label}</span></li>`;
      })
      .join("");
    els.status.innerHTML = `<ol class="status-steps">${items}</ol>`;
  }

  function prettyType(type) {
    return TYPE_LABELS[type] || String(type || "region").replaceAll("_", " ");
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Could not decode image."));
      img.src = src;
    });
  }

  function drawImage(canvas, img) {
    const ctx = canvas.getContext("2d");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
  }

  function drawOverlay() {
    if (!state.images.screenshot || !state.images.heatmap) return;
    const canvas = els.canvasOverlay;
    const shot = state.images.screenshot;
    const heat = state.images.heatmap;
    const ctx = canvas.getContext("2d");
    canvas.width = shot.naturalWidth || shot.width;
    canvas.height = shot.naturalHeight || shot.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(shot, 0, 0);
    ctx.globalAlpha = Number(els.opacity.value) / 100;
    ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;

    if (els.showBoxes.checked && state.payload?.regions) {
      for (const region of state.payload.regions) {
        ctx.strokeStyle = TYPE_COLORS[region.element_type] || "#ffffff";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.55;
        ctx.strokeRect(region.x, region.y, region.width, region.height);
      }
      ctx.globalAlpha = 1;
    }

    const hotspots = state.payload?.hotspots || [];
    if (els.showHotspots.checked && hotspots.length) {
      hotspots.forEach((h, i) => {
        const active = state.activeHotspot === i;
        const color = TYPE_COLORS[h.element_type] || "#0f6b5c";
        ctx.strokeStyle = color;
        ctx.lineWidth = active ? 3 : 2;
        ctx.strokeRect(h.x + 0.5, h.y + 0.5, h.width, h.height);
        if (active) {
          ctx.fillStyle = "rgba(15, 107, 92, 0.18)";
          ctx.fillRect(h.x, h.y, h.width, h.height);
        }
        const badge = 20;
        const bx = Math.min(h.x + 4, canvas.width - badge - 2);
        const by = Math.max(2, h.y - 2);
        ctx.fillStyle = color;
        ctx.fillRect(bx, by, badge, badge);
        ctx.fillStyle = "#ffffff";
        ctx.font = "700 12px Figtree, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), bx + badge / 2, by + badge / 2 + 0.5);
      });
    }
  }

  function selectHotspot(index) {
    state.activeHotspot = index;
    [...els.hotspots.querySelectorAll(".cell")].forEach((cell, i) => {
      cell.classList.toggle("active", i === index);
    });
    drawOverlay();
  }

  function renderHotspots(hotspots) {
    if (!hotspots?.length) {
      els.hotspots.innerHTML =
        `<div class="cell empty"><span class="label">No distinct hot zones found</span></div>`;
      return;
    }
    els.hotspots.innerHTML = hotspots
      .map((h, i) => {
        const pct = Math.round((h.relative ?? h.score) * 100);
        const zone = (h.zone || "region").replaceAll("-", " ");
        return `
        <button type="button" class="cell" role="listitem" data-index="${i}" data-type="${h.element_type}">
          <div class="cell-top">
            <span class="rank">${i + 1}</span>
            <span class="score">${pct}<small>%</small></span>
          </div>
          <span class="label">${prettyType(h.element_type)}</span>
          <span class="zone">${zone}</span>
          <span class="meta-line">${h.area_pct ?? "—"}% of page</span>
          <div class="bar" aria-hidden="true"><span style="width:${pct}%"></span></div>
        </button>`;
      })
      .join("");

    [...els.hotspots.querySelectorAll(".cell[data-index]")].forEach((cell) => {
      cell.addEventListener("click", () => {
        const idx = Number(cell.dataset.index);
        selectHotspot(state.activeHotspot === idx ? -1 : idx);
      });
    });
  }

  function renderStructure(counts) {
    const entries = Object.entries(counts || {});
    if (!entries.length) {
      els.structureCounts.innerHTML = `<span>No regions</span>`;
      return;
    }
    els.structureCounts.innerHTML = entries
      .map(
        ([type, count]) =>
          `<span data-type="${type}">${prettyType(type)} · ${count}</span>`
      )
      .join("");
  }

  async function showResult(payload) {
    state.payload = payload;
    state.activeHotspot = -1;
    const [shot, heat] = await Promise.all([
      loadImage(payload.screenshot),
      loadImage(payload.heatmap),
    ]);
    state.images.screenshot = shot;
    state.images.heatmap = heat;

    drawImage(els.canvasOriginal, shot);
    drawImage(els.canvasHeatmap, heat);
    drawOverlay();

    els.modelBadge.textContent = `Model · ${payload.model || "—"}`;
    const sourceMap = {
      upload: "Screenshot",
      sample: "Sample page",
      url: "Live URL + DOM",
    };
    const modeLabel =
      payload.structure_mode === "live_dom" ? "live DOM" : "image→DOM";
    els.sourceBadge.textContent = `${sourceMap[payload.source] || payload.source || "predict"} · ${modeLabel}`;

    renderHotspots(payload.hotspots);
    renderStructure(payload.region_counts);
    els.domHtml.textContent = payload.html_dom || "(empty)";

    els.emptyState.classList.add("hidden");
    els.results.classList.remove("hidden");
    setWorkflow("output");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function setPreview(file) {
    clearPreviewUrl();
    state.file = file;
    if (!file) {
      els.previewWrap.classList.add("hidden");
      return;
    }
    state.previewUrl = URL.createObjectURL(file);
    els.preview.src = state.previewUrl;
    els.previewWrap.classList.remove("hidden");
    setWorkflow("input");
  }

  function clearPreviewUrl() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
    state.file = null;
    els.file.value = "";
    els.preview.removeAttribute("src");
    els.previewWrap.classList.add("hidden");
  }

  async function predictUpload() {
    if (!state.file) {
      setStatusError("Choose a screenshot first.");
      return;
    }
    setBusy(true);
    setWorkflow("run");
    setStatusSteps(["Reading image", "Building structure", "Predicting attention"], 0);
    const body = new FormData();
    body.append("file", state.file);
    try {
      setStatusSteps(["Reading image", "Building structure", "Predicting attention"], 1);
      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(detailMessage(data) || "Prediction failed.");
      setStatusSteps(["Reading image", "Building structure", "Predicting attention"], 3);
      await showResult(data);
      setStatusIdle("Done — explore the overlay and hot zones.");
    } catch (err) {
      setStatusError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  async function predictUrl() {
    const url = (els.url.value || "").trim();
    if (!url) {
      setStatusError("Enter a full http or https URL.");
      return;
    }
    setBusy(true);
    setWorkflow("run");
    setStatusSteps(["Capturing page", "Reading DOM", "Predicting attention"], 0);
    const body = new FormData();
    body.append("url", url);
    try {
      const advance = window.setTimeout(() => {
        setStatusSteps(["Capturing page", "Reading DOM", "Predicting attention"], 1);
      }, 1200);
      const res = await fetch("/api/predict", { method: "POST", body });
      window.clearTimeout(advance);
      setStatusSteps(["Capturing page", "Reading DOM", "Predicting attention"], 2);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(detailMessage(data) || "Prediction failed.");
      setStatusSteps(["Capturing page", "Reading DOM", "Predicting attention"], 3);
      await showResult(data);
      setStatusIdle("Done — live DOM structure included.");
    } catch (err) {
      setStatusError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  async function predictSample() {
    setMode("screenshot");
    setBusy(true);
    setWorkflow("run");
    setStatusSteps(["Loading sample", "Building structure", "Predicting attention"], 0);
    try {
      setStatusSteps(["Loading sample", "Building structure", "Predicting attention"], 1);
      const res = await fetch("/api/sample");
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(detailMessage(data) || "Sample failed.");
      setStatusSteps(["Loading sample", "Building structure", "Predicting attention"], 3);
      await showResult(data);
      setStatusIdle("Done — sample heatmap ready.");
    } catch (err) {
      setStatusError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  function detailMessage(data) {
    if (!data || data.detail == null) return "";
    if (typeof data.detail === "string") return data.detail;
    return JSON.stringify(data.detail);
  }

  function downloadCanvas(canvas, filename) {
    const link = document.createElement("a");
    link.download = filename;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  async function loadInfo() {
    try {
      const res = await fetch("/api/info");
      const info = await res.json();
      if (!res.ok) throw new Error("info failed");
      const m = info.metrics || {};
      const model = info.model || m.best_model || "—";
      const cc = m.CC != null ? Number(m.CC).toFixed(4) : "—";
      const auc = m["AUC-Judd"] != null ? Number(m["AUC-Judd"]).toFixed(4) : "—";
      const n = info.fiwi_images != null ? info.fiwi_images : "—";
      const ck = info.checkpoint ? "checkpoint ready" : "checkpoint missing";
      els.footerMetrics.innerHTML =
        `<strong>WebGaze</strong> · best model <strong>${model}</strong> · FiWi CC ${cc}, AUC-Judd ${auc} (${n} images) · ${ck}`;
      if (!info.sample_ready) {
        els.sampleBtn.disabled = true;
        els.sampleBtn.title = "Sample image not found";
      }
    } catch {
      els.footerMetrics.textContent =
        "WebGaze · model metrics unavailable (is the API running?)";
    }
  }

  els.modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  });

  els.dropzone.addEventListener("click", () => els.file.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.file.click();
    }
  });

  ["dragenter", "dragover"].forEach((type) => {
    els.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    els.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) setPreview(file);
  });

  els.file.addEventListener("change", () => {
    const file = els.file.files?.[0];
    if (file) setPreview(file);
  });

  els.clearPreview.addEventListener("click", () => {
    clearPreviewUrl();
    setStatusIdle("Ready when you are.");
    setWorkflow("input");
  });

  els.predictBtn.addEventListener("click", () => {
    if (state.busy) return;
    if (state.mode === "url") predictUrl();
    else predictUpload();
  });

  els.sampleBtn.addEventListener("click", () => {
    if (!state.busy) predictSample();
  });

  els.opacity.addEventListener("input", drawOverlay);
  els.showBoxes.addEventListener("change", drawOverlay);
  els.showHotspots.addEventListener("change", drawOverlay);

  els.domToggle.addEventListener("click", () => {
    const open = els.domPanel.classList.toggle("hidden") === false;
    els.domToggle.setAttribute("aria-expanded", open ? "true" : "false");
    els.domToggle.querySelector(".chevron").textContent = open ? "−" : "+";
  });

  els.dlOverlay.addEventListener("click", () => {
    downloadCanvas(els.canvasOverlay, "webgaze-overlay.png");
  });
  els.dlHeatmap.addEventListener("click", () => {
    downloadCanvas(els.canvasHeatmap, "webgaze-heatmap.png");
  });

  document.querySelectorAll(".step[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      const target = document.getElementById(btn.dataset.jump);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  setMode("screenshot");
  setWorkflow("input");
  setStatusIdle("Ready when you are.");
  loadInfo();
})();

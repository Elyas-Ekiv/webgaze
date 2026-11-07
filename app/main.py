"""WebGaze demo API: screenshot or URL → saliency heatmap."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import IMG_H, IMG_W, RESULTS  # noqa: E402
from src.data.html_dom import boxes_to_dom, image_to_dom  # noqa: E402
from src.infer import heatmap_to_color, load_checkpoint, predict_bgr  # noqa: E402

STATIC = Path(__file__).parent / "static"
EXAMPLE = STATIC / "examples" / "sample.jpg"

app = FastAPI(title="WebGaze")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache: dict = {}

DOM_SCRIPT = """() => {
  const sel = 'header,nav,h1,h2,h3,p,a,button,img,input,li,section,article';
  const pageArea = window.innerWidth * window.innerHeight;
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8 || r.width * r.height > 0.55 * pageArea) continue;
    const tag = el.tagName.toLowerCase();
    let element_type = 'layout_region';
    if (['header','nav','h1','h2'].includes(tag)) element_type = 'header';
    else if (['p','li','h3','a','button','input','label'].includes(tag)) element_type = 'text_or_content';
    out.push({
      x: r.left, y: r.top, width: r.width, height: r.height,
      element_type,
      text_length: (el.innerText || '').trim().length,
      font_size: parseFloat(getComputedStyle(el).fontSize) || 0,
    });
    if (out.length >= 48) break;
  }
  return { width: window.innerWidth, height: window.innerHeight, boxes: out };
}"""


def get_model():
    if "model" not in _cache:
        _cache["model"], _cache["device"], _cache["name"] = load_checkpoint()
    return _cache["model"], _cache["device"], _cache["name"]


def _b64_png(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _overlay(frame_bgr: np.ndarray, heatmap_color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_color, alpha, 0)


def _zone_name(cx: float, cy: float, w: int, h: int) -> str:
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    row = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}"


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return float(inter / union) if union else 0.0


def _hotspots(heatmap: np.ndarray, regions: list[dict], top_n: int = 5) -> list[dict]:
    """Rank distinct layout regions by mean saliency (dedupe overlaps)."""
    h, w = heatmap.shape[:2]
    page = float(max(1, h * w))
    scored: list[dict] = []
    for region in regions:
        x = int(np.clip(region["x"], 0, w - 1))
        y = int(np.clip(region["y"], 0, h - 1))
        rw = int(np.clip(region["width"], 1, w - x))
        rh = int(np.clip(region["height"], 1, h - y))
        if rw * rh < 24:
            continue
        patch = heatmap[y : y + rh, x : x + rw]
        if patch.size == 0:
            continue
        mean = float(patch.mean())
        peak = float(patch.max())
        scored.append(
            {
                "element_type": region["element_type"],
                "x": x,
                "y": y,
                "width": rw,
                "height": rh,
                "depth": float(region.get("depth", 0)),
                "score": round(mean, 4),
                "peak": round(peak, 4),
                "area_pct": round(100.0 * rw * rh / page, 1),
                "zone": _zone_name(x + rw / 2, y + rh / 2, w, h),
            }
        )
    scored.sort(key=lambda item: (item["score"], item["peak"]), reverse=True)

    picked: list[dict] = []
    for item in scored:
        if any(_iou(item, kept) >= 0.45 for kept in picked):
            continue
        picked.append(item)
        if len(picked) >= top_n:
            break

    if picked:
        top = picked[0]["score"] or 1.0
        for item in picked:
            item["relative"] = round(item["score"] / top, 3)
    return picked


def _region_counts(regions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        key = region.get("element_type", "layout_region")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _scale_box(box: dict, sx: float, sy: float) -> dict:
    return {
        **box,
        "x": int(round(box["x"] * sx)),
        "y": int(round(box["y"] * sy)),
        "width": max(1, int(round(box["width"] * sx))),
        "height": max(1, int(round(box["height"] * sy))),
    }


def _predict(image_bgr: np.ndarray, boxes: list[dict] | None = None, source: str = "upload") -> dict:
    model, device, name = get_model()
    heatmap, regions = predict_bgr(model, image_bgr, device, boxes)
    color = heatmap_to_color(heatmap)
    frame = cv2.resize(image_bgr, (color.shape[1], color.shape[0]))
    _, _, _, html = boxes_to_dom(frame, boxes) if boxes else image_to_dom(frame)

    model_regions = [
        {
            "element_type": r["element_type"],
            "x": r["x"],
            "y": r["y"],
            "width": r["width"],
            "height": r["height"],
            "depth": r.get("depth", 0),
        }
        for r in regions
    ]
    hotspots = _hotspots(heatmap, model_regions)

    # Upscale demo visuals for clearer UI (model still runs at native size).
    display_scale = 2
    disp_w, disp_h = color.shape[1] * display_scale, color.shape[0] * display_scale
    frame_hi = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)
    color_hi = cv2.resize(color, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)
    overlay = _overlay(frame_hi, color_hi, alpha=0.5)
    region_payload = [_scale_box(r, display_scale, display_scale) for r in model_regions]
    hotspot_payload = [_scale_box(h, display_scale, display_scale) for h in hotspots]

    return {
        "model": name,
        "source": source,
        "screenshot": _b64_png(frame_hi),
        "heatmap": _b64_png(color_hi),
        "overlay": _b64_png(overlay),
        "html_dom": html,
        "regions": region_payload,
        "region_counts": _region_counts(region_payload),
        "hotspots": hotspot_payload,
        "structure_mode": "live_dom" if boxes else "image_to_dom",
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "checkpoint": (ROOT / "models" / "best_multimodal_model.pth").exists()}


@app.get("/api/info")
def info():
    checkpoint = ROOT / "models" / "best_multimodal_model.pth"
    metrics_path = RESULTS / "metrics.json"
    payload = {
        "brand": "WebGaze",
        "checkpoint": checkpoint.exists(),
        "model": None,
        "fiwi_images": None,
        "metrics": None,
        "sample_ready": EXAMPLE.exists(),
    }
    if checkpoint.exists():
        try:
            _, _, name = get_model()
            payload["model"] = name
        except Exception:
            payload["model"] = "unavailable"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            best = data.get("best_model")
            table = data.get("fiwi", {})
            best_scores = table.get(best, {}) if best else {}
            payload["fiwi_images"] = data.get("fiwi_images")
            payload["metrics"] = {
                "best_model": best,
                "CC": round(float(best_scores.get("CC", 0)), 4) if best_scores else None,
                "AUC-Judd": round(float(best_scores.get("AUC-Judd", 0)), 4) if best_scores else None,
                "NSS": round(float(best_scores.get("NSS", 0)), 4) if best_scores else None,
                "SIM": round(float(best_scores.get("SIM", 0)), 4) if best_scores else None,
                "KL-Divergence": round(float(best_scores.get("KL-Divergence", 0)), 4) if best_scores else None,
            }
            if not payload["model"] and best:
                payload["model"] = best
        except Exception:
            pass
    return payload


@app.post("/api/predict")
async def predict(file: UploadFile | None = File(default=None), url: str = Form(default="")):
    url = (url or "").strip()
    if url:
        return await _from_url(url)
    if file is None:
        raise HTTPException(400, "Upload a screenshot or enter a URL.")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image is larger than 15 MB.")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "That file is not a readable image.")
    try:
        return _predict(image, source="upload")
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/sample")
def sample():
    if not EXAMPLE.exists():
        raise HTTPException(404, "Sample image is not ready yet.")
    try:
        return _predict(cv2.imread(str(EXAMPLE), cv2.IMREAD_COLOR), source="sample")
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


async def _from_url(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "Enter a full http or https URL.")
    try:
        image, boxes = await _capture(url)
        return _predict(image, boxes, source="url")
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not open that page: {exc}") from exc


async def _capture(url: str) -> tuple[np.ndarray, list[dict]]:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1360, "height": 768})
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1200)
        png = await page.screenshot(full_page=False)
        dom = await page.evaluate(DOM_SCRIPT)
        await browser.close()

    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Screenshot failed")

    h, w = image.shape[:2]
    sx, sy = IMG_W / w, IMG_H / h
    boxes = [
        {
            "x": int(b["x"] * sx),
            "y": int(b["y"] * sy),
            "width": max(1, int(b["width"] * sx)),
            "height": max(1, int(b["height"] * sy)),
            "element_type": b["element_type"],
            "text_length": b["text_length"],
            "font_size": b["font_size"],
        }
        for b in dom["boxes"]
    ]
    return image, boxes

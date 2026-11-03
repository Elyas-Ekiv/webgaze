"""Demo API: screenshot upload or live URL → saliency heatmap."""

from __future__ import annotations

import base64
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

from src.config import IMG_H, IMG_W  # noqa: E402
from src.data.html_dom import boxes_to_dom, image_to_dom  # noqa: E402
from src.infer import heatmap_to_color, load_checkpoint, predict_bgr  # noqa: E402

STATIC = Path(__file__).parent / "static"
EXAMPLE = STATIC / "examples" / "sample.jpg"

app = FastAPI(title="Webpage Attention")
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


def _predict(image_bgr: np.ndarray, boxes: list[dict] | None = None) -> dict:
    model, device, name = get_model()
    heatmap, regions = predict_bgr(model, image_bgr, device, boxes)
    color = heatmap_to_color(heatmap)
    frame = cv2.resize(image_bgr, (color.shape[1], color.shape[0]))
    _, _, _, html = boxes_to_dom(frame, boxes) if boxes else image_to_dom(frame)
    return {
        "model": name,
        "screenshot": _b64_png(frame),
        "heatmap": _b64_png(color),
        "html_dom": html,
        "regions": [
            {
                "element_type": r["element_type"],
                "x": r["x"],
                "y": r["y"],
                "width": r["width"],
                "height": r["height"],
                "depth": r.get("depth", 0),
            }
            for r in regions
        ],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "checkpoint": (ROOT / "models" / "best_multimodal_model.pth").exists()}


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
        return _predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/sample")
def sample():
    if not EXAMPLE.exists():
        raise HTTPException(404, "Sample image is not ready yet.")
    try:
        return _predict(cv2.imread(str(EXAMPLE), cv2.IMREAD_COLOR))
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


async def _from_url(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "Enter a full http or https URL.")
    try:
        image, boxes = await _capture(url)
        return _predict(image, boxes)
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

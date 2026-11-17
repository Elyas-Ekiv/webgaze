from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import IMG_H, IMG_W, MODELS_DIR
from src.data.structural import extract_prototype, features_from_boxes
from src.models.multimodal import build_model


def load_checkpoint(path: Path | None = None, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = path or (MODELS_DIR / "best_multimodal_model.pth")
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    note = ckpt.get("note") if isinstance(ckpt, dict) else None
    return model, device, ckpt["model_name"], (str(note) if note else None)


@torch.no_grad()
def predict_bgr(model, image_bgr: np.ndarray, device: torch.device, boxes: list[dict] | None = None):
    resized = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    features, mask, regions = features_from_boxes(resized, boxes) if boxes else extract_prototype(resized)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    pred = model(
        image.to(device),
        torch.from_numpy(features).unsqueeze(0).to(device),
        torch.from_numpy(mask).unsqueeze(0).to(device),
    )
    return np.clip(pred[0, 0].float().cpu().numpy(), 0.0, 1.0), regions


def _stretch_range(heatmap: np.ndarray, lo: float, hi: float) -> np.ndarray | None:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return None
    return np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)


def display_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Percentile-stretch a saliency map for visualization only.

    Metrics keep the raw prediction. An initialized decoder can sit near 0.5
    with almost no range; stretching the 2nd–98th percentiles makes the spatial
    variation visible without changing stored scores or checkpoint weights.
    A tiny percentile span falls back to min-max. A constant map stays flat.
    Do not run this on a layout demo map — that map is already 0..1.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    stretched = _stretch_range(h, float(np.percentile(h, 2)), float(np.percentile(h, 98)))
    if stretched is None:
        stretched = _stretch_range(h, float(np.min(h)), float(np.max(h)))
    if stretched is None:
        return np.clip(h, 0.0, 1.0)
    return stretched


def is_initialized_demo(note: str | None) -> bool:
    """True when the checkpoint note says these are untrained demo weights."""
    if not note:
        return False
    text = note.lower()
    return "initialized" in text and ("demo" in text or "not the trained" in text)


def heatmap_range_tiny(heatmap: np.ndarray, eps: float = 0.05) -> bool:
    """True when a prediction has almost no spatial range (noise around 0.5)."""
    h = np.asarray(heatmap, dtype=np.float32)
    finite = h[np.isfinite(h)]
    if finite.size == 0:
        return True
    return float(finite.max() - finite.min()) < eps


def _clip_box(region: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    x = int(np.clip(region.get("x", 0), 0, max(width - 1, 0)))
    y = int(np.clip(region.get("y", 0), 0, max(height - 1, 0)))
    rw = int(np.clip(region.get("width", 1), 1, max(width - x, 1)))
    rh = int(np.clip(region.get("height", 1), 1, max(height - y, 1)))
    if rw < 2 or rh < 2 or x >= width or y >= height:
        return None
    return x, y, rw, rh


def _coverage(box: tuple[int, int, int, int], regions: list[dict]) -> float:
    x, y, w, h = box
    mask = np.zeros((h, w), np.uint8)
    for region in regions:
        rx, ry = int(region.get("x", 0)), int(region.get("y", 0))
        rw, rh = int(region.get("width", 0)), int(region.get("height", 0))
        ix1, iy1 = max(x, rx), max(y, ry)
        ix2, iy2 = min(x + w, rx + rw), min(y + h, ry + rh)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1 - y : iy2 - y, ix1 - x : ix2 - x] = 1
    return float(mask.mean()) if mask.size else 1.0


def _demo_weight(region: dict, width: int, height: int) -> float:
    """Hot on headers, buttons, images, and short top text. Warm on other copy."""
    element_type = str(region.get("element_type") or "layout_region")
    role = str(region.get("demo_role") or "")
    if element_type == "background" or role == "background":
        return 0.0
    box = _clip_box(region, width, height)
    if box is None:
        return 0.0
    x, y, rw, rh = box
    page = float(max(width * height, 1))
    area = (rw * rh) / page
    aspect = rw / max(rh, 1)
    cy = y + rh / 2.0
    short = rh <= max(16, 0.08 * height) and rw <= 0.7 * width
    image_like = 0.35 <= aspect <= 1.75 and rh >= 18 and 0.006 <= area <= 0.12
    body = area >= 0.04 and aspect >= 1.55 and rh >= 26
    upper = cy < 0.50 * height

    if role == "button" or element_type == "header":
        weight = 1.0
    elif element_type == "layout_region":
        weight = 0.92 if image_like else 0.34
    elif element_type == "text_or_content":
        if short and (upper or cy < 0.72 * height):
            weight = 1.0 if upper else 0.78
        elif image_like:
            weight = 0.96
        elif body and upper:
            weight = 0.9
        elif body:
            weight = 0.58
        else:
            weight = 0.84
    else:
        weight = 0.4

    text_length = float(region.get("text_length") or 0)
    if text_length > 200 and element_type == "text_or_content":
        weight = min(weight, 0.6)
    elif 0 < text_length <= 36 and element_type in {"header", "text_or_content"}:
        weight = max(weight, 0.96)
    font_size = float(region.get("font_size") or 0)
    if font_size >= 22 and element_type in {"header", "text_or_content"}:
        weight = max(weight, 0.94)
    if area > 0.2 and element_type != "header":
        weight *= 0.4
    return float(weight)


def _extra_demo_regions(image_bgr: np.ndarray, regions: list[dict], shape: tuple[int, int]) -> list[dict]:
    """Content boxes the feature dedupe drops, plus saturated buttons.

    The model region list keeps inner strips and throws away the parent card,
    so a search form and its button would otherwise stay cold. These boxes are
    for the demo map only. They do not change checkpoint features.
    """
    from src.data.structural import _appearance, _classify

    height, width = shape
    frame = image_bgr
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    closed = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page = float(width * height)
    min_area = 0.004 * page
    max_area = 0.18 * page
    extra: list[dict] = []
    seen = list(regions)
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        if rw < 10 or rh < 10 or not (min_area <= rw * rh <= max_area):
            continue
        edge_density, non_white, contrast, brightness = _appearance(gray, edges, (x, y, rw, rh))
        element_type = _classify(y, rh, height, edge_density, non_white, rw, width, contrast, brightness)
        if element_type == "background":
            continue
        if _coverage((x, y, rw, rh), seen) >= 0.42:
            continue
        box = {
            "x": int(x),
            "y": int(y),
            "width": int(rw),
            "height": int(rh),
            "element_type": element_type,
            "text_length": 0.0,
            "font_size": 0.0,
            "demo_role": "recovered",
        }
        extra.append(box)
        seen.append(box)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturated = ((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 110)).astype(np.uint8) * 255
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    button_contours, _ = cv2.findContours(saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in button_contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        if rw < 28 or rh < 10 or rh > 0.14 * height or rw > 0.4 * width:
            continue
        if rw * rh < 500 or rw < rh * 1.4:
            continue
        # A parent card can cover a button. Only skip when a similar control is already boxed.
        similar = [
            region
            for region in seen
            if int(region.get("width", 0)) * int(region.get("height", 0)) <= int(2.5 * rw * rh)
        ]
        if similar and _coverage((x, y, rw, rh), similar) >= 0.55:
            continue
        box = {
            "x": int(x),
            "y": int(y),
            "width": int(rw),
            "height": int(rh),
            "element_type": "text_or_content",
            "text_length": 12.0,
            "font_size": 16.0,
            "demo_role": "button",
        }
        extra.append(box)
        seen.append(box)
    return extra


def demo_attention_regions(
    regions: list[dict] | None,
    image_bgr: np.ndarray | None,
    shape: tuple[int, ...],
) -> list[dict]:
    height = int(shape[0])
    width = int(shape[1])
    kept = [dict(region) for region in (regions or [])]
    if image_bgr is None:
        return kept
    kept.extend(_extra_demo_regions(image_bgr, kept, (height, width)))
    return kept


def layout_demo_heatmap(
    regions: list[dict] | None,
    shape: tuple[int, ...],
    image_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth gaze-like map from layout boxes. Not a trained FiWi prediction.

    Header, buttons, links, images, and short text near the top and center are
    hot. Other text is warm. Empty background stays near 0. Each box is a
    Gaussian blob (not a hard rectangle), scaled by a mild top-and-center prior.
    The result is normalized to 0..1 and is not percentile-stretched.
    """
    height = int(shape[0]) if len(shape) >= 1 else 0
    width = int(shape[1]) if len(shape) >= 2 else 0
    heat = np.zeros((height, width), dtype=np.float32)
    if height < 2 or width < 2:
        return heat
    boxes = demo_attention_regions(regions, image_bgr, (height, width))
    if not boxes:
        return heat

    ys = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    xs = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    # Mild prior: equal boxes prefer the top middle, empty pixels stay 0.
    prior_x = np.exp(-0.5 * ((xs - 0.50) / 0.60) ** 2)
    prior_y = np.exp(-0.5 * ((ys - 0.20) / 0.52) ** 2)
    prior = prior_y[:, None] * prior_x[None, :]
    prior_max = float(prior.max())
    if prior_max > 0:
        prior = 0.82 + 0.18 * (prior / prior_max)

    for region in boxes:
        weight = _demo_weight(region, width, height)
        if weight <= 0:
            continue
        clipped = _clip_box(region, width, height)
        if clipped is None:
            continue
        x, y, rw, rh = clipped
        cx = x + rw / 2.0
        cy = y + rh / 2.0
        sigma_x = max(3.2, rw * 0.36)
        sigma_y = max(2.6, rh * 0.40)
        x0 = max(0, int(np.floor(cx - 2.6 * sigma_x)))
        x1 = min(width, int(np.ceil(cx + 2.6 * sigma_x)) + 1)
        y0 = max(0, int(np.floor(cy - 2.6 * sigma_y)))
        y1 = min(height, int(np.ceil(cy + 2.6 * sigma_y)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        grid_y = np.arange(y0, y1, dtype=np.float32)[:, None]
        grid_x = np.arange(x0, x1, dtype=np.float32)[None, :]
        blob = np.exp(-0.5 * ((grid_x - cx) / sigma_x) ** 2 - 0.5 * ((grid_y - cy) / sigma_y) ** 2)
        # Max, not sum: overlapping strips must not stack into a fake peak.
        heat[y0:y1, x0:x1] = np.maximum(
            heat[y0:y1, x0:x1],
            np.float32(weight) * blob * prior[y0:y1, x0:x1],
        )

    peak = float(heat.max()) if heat.size else 0.0
    if peak < 1e-8:
        return heat
    heat /= np.float32(peak)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=1.05, sigmaY=1.05)
    peak = float(heat.max())
    if peak > 1e-8:
        heat /= np.float32(peak)
    return np.clip(heat, 0.0, 1.0)


def heatmap_for_display(
    heatmap: np.ndarray,
    regions: list[dict] | None,
    checkpoint_note: str | None,
    image_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, str, list[dict]]:
    """Pick the map shown in the demo.

    Initialized demo weights, and any prediction whose raw range is tiny, are
    drawn from layout regions. That map is not percentile-stretched. A trained
    checkpoint with real range still uses the model output.
    """
    raw = np.asarray(heatmap, dtype=np.float32)
    source = [dict(region) for region in (regions or [])]
    if is_initialized_demo(checkpoint_note) or heatmap_range_tiny(raw):
        boxes = demo_attention_regions(source, image_bgr, raw.shape)
        return layout_demo_heatmap(source, raw.shape, image_bgr), "layout_demo", boxes
    return display_heatmap(raw), "contrast_stretched", source


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Color a map that is already scaled to 0..1. Does not stretch."""
    vis = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    return cv2.applyColorMap(np.clip(vis * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    vis = display_heatmap(heatmap)
    return colorize_heatmap(vis)

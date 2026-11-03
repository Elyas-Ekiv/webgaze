"""Find / normalize layout boxes. Feature packing lives in html_dom.py."""

from __future__ import annotations

import numpy as np

from src.config import ELEMENT_TYPES, IMG_H, IMG_W, MAX_REGIONS

_TYPE_INDEX = {name: i for i, name in enumerate(ELEMENT_TYPES)}


def _appearance(gray: np.ndarray, edges: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    crop = gray[y : y + h, x : x + w]
    edge = edges[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0, 0.0, 0.0, 1.0
    edge_density = float(edge.mean()) / 255.0
    non_white = float((crop < 245).mean())
    contrast = float(crop.std()) / 128.0
    brightness = float(crop.mean()) / 255.0
    return edge_density, non_white, contrast, brightness


def _classify(
    y: int,
    h: int,
    image_h: int,
    edge_density: float,
    non_white: float,
    w: int,
    image_w: int,
    contrast: float = 0.0,
    brightness: float = 1.0,
) -> str:
    """Map a box onto the four region types used by the structural branch."""
    center_y = (y + h / 2.0) / max(image_h, 1)
    width_frac = w / max(image_w, 1)
    height_frac = h / max(image_h, 1)
    text_like = edge_density >= 0.045 or (contrast >= 0.16 and non_white >= 0.2 and edge_density >= 0.02)
    near_white = brightness >= 0.93 and non_white < 0.18 and contrast < 0.08
    flat_panel = edge_density < 0.055 and contrast < 0.11
    if center_y < 0.22 and height_frac < 0.40 and width_frac >= 0.28 and (text_like or not near_white):
        return "header"
    if flat_panel and near_white:
        return "background"
    if flat_panel:
        return "layout_region"
    if text_like and height_frac < 0.7:
        return "text_or_content"
    if near_white:
        return "background"
    return "layout_region"


def _containment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller else 0.0


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _panel_boxes(gray: np.ndarray, edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Group nearby content cells so a flat card becomes one box, not an empty grid."""
    import cv2

    cell = 16
    rows = IMG_H // cell
    cols = IMG_W // cell
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            y, x = row * cell, col * cell
            patch_g = gray[y : y + cell, x : x + cell]
            patch_e = edges[y : y + cell, x : x + cell]
            brightness = float(patch_g.mean()) / 255.0
            contrast = float(patch_g.std()) / 128.0
            edge_density = float(patch_e.mean()) / 255.0
            if edge_density >= 0.02 or contrast >= 0.08 or brightness < 0.90:
                mask[row, col] = 255
    count, labels = cv2.connectedComponents(mask, connectivity=4)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs) < 2:
            continue
        x0 = int(xs.min() * cell)
        y0 = int(ys.min() * cell)
        x1 = int((xs.max() + 1) * cell)
        y1 = int((ys.max() + 1) * cell)
        w, h = x1 - x0, y1 - y0
        if w < 16 or h < 12 or w * h > 0.72 * IMG_W * IMG_H:
            continue
        boxes.append((x0, y0, w, h))
    return boxes


def _dedupe(regions: list[dict]) -> list[dict]:
    ranked = sorted(regions, key=lambda r: (-r.get("text_length", 0.0), r["width"] * r["height"]))
    kept: list[dict] = []
    for region in ranked:
        box = (region["x"], region["y"], region["width"], region["height"])
        if all(
            _iou(box, (other["x"], other["y"], other["width"], other["height"])) < 0.55
            and _containment(box, (other["x"], other["y"], other["width"], other["height"])) < 0.72
            for other in kept
        ):
            kept.append(region)
        if len(kept) >= MAX_REGIONS:
            break
    kept.sort(key=lambda r: (r["y"], r["x"]))
    return kept


def detect_regions(image_bgr: np.ndarray) -> list[dict]:
    """Estimate layout regions from a screenshot with OpenCV (image-to-DOM front-end)."""
    import cv2

    if image_bgr.ndim != 3:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    min_area = 0.004 * IMG_W * IMG_H
    max_area = 0.45 * IMG_W * IMG_H
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        if not (min_area <= w * h <= max_area):
            continue
        candidates.append((x, y, w, h))

    candidates.extend(_panel_boxes(gray, edges))
    candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in candidates:
        if all(_iou(box, other) < 0.55 for other in kept):
            kept.append(box)
        if len(kept) >= MAX_REGIONS:
            break

    if not kept:
        kept.append((0, 0, IMG_W, IMG_H))

    regions: list[dict] = []
    for x, y, w, h in kept[:MAX_REGIONS]:
        edge_density, non_white, contrast, brightness = _appearance(gray, edges, (x, y, w, h))
        element_type = _classify(y, h, IMG_H, edge_density, non_white, w, IMG_W, contrast, brightness)
        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "element_type": element_type,
                "text_length": 0.0,
                "font_size": 0.0,
            }
        )

    content = [region for region in regions if region["element_type"] != "background"]
    return _dedupe(content or regions[:1])


def normalize_dom_boxes(image_bgr: np.ndarray, boxes: list[dict]) -> list[dict]:
    """Normalize Playwright/DOM boxes onto the model canvas."""
    import cv2

    image_bgr = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    regions = []
    page_area = IMG_W * IMG_H
    for box in boxes:
        w = int(box.get("width", 0))
        h = int(box.get("height", 0))
        if w <= 1 or h <= 1 or w * h > 0.55 * page_area:
            continue
        x = int(np.clip(box.get("x", 0), 0, IMG_W - 1))
        y = int(np.clip(box.get("y", 0), 0, IMG_H - 1))
        w = int(np.clip(w, 1, IMG_W - x))
        h = int(np.clip(h, 1, IMG_H - y))
        element_type = box.get("element_type", "layout_region")
        if element_type not in _TYPE_INDEX:
            element_type = "layout_region"
        edge_density, non_white, contrast, brightness = _appearance(gray, edges, (x, y, w, h))
        visual = _classify(y, h, IMG_H, edge_density, non_white, w, IMG_W, contrast, brightness)
        text_length = float(box.get("text_length", 0.0))
        if element_type == "layout_region":
            element_type = visual
        elif element_type != "header" and visual == "background" and text_length < 8:
            element_type = "background"
        regions.append(
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "element_type": element_type,
                "text_length": text_length,
                "font_size": float(box.get("font_size", 0.0)),
            }
        )
    if not regions:
        return detect_regions(image_bgr)
    content = [region for region in regions if region["element_type"] != "background"]
    return _dedupe(content or regions[:1])


def extract_prototype(image_bgr: np.ndarray):
    """Backward-compatible API: features, mask, regions via image-to-DOM."""
    from src.data.html_dom import image_to_dom

    features, mask, regions, _html = image_to_dom(image_bgr)
    return features, mask, regions


def features_from_boxes(image_bgr: np.ndarray, boxes: list[dict]):
    """Backward-compatible API for live DOM boxes."""
    from src.data.html_dom import boxes_to_dom

    features, mask, regions, _html = boxes_to_dom(image_bgr, boxes)
    return features, mask, regions


def feature_names() -> list[str]:
    from src.data.html_dom import feature_names as names

    return names()

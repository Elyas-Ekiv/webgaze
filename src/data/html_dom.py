"""Turn layout regions into the structural feature tensor (+ a tiny HTML dump)."""

from __future__ import annotations

import numpy as np

from src.config import ELEMENT_TYPES, IMG_H, IMG_W, MAX_REGIONS, NUMERIC_FEATURES, STRUCT_DIM
from src.data.structural import _dedupe, detect_regions, normalize_dom_boxes

_TYPE_INDEX = {name: i for i, name in enumerate(ELEMENT_TYPES)}
_TAG = {"header": "header", "text_or_content": "p", "background": "section", "layout_region": "div"}


def _clip01(v: float) -> float:
    return 0.0 if not np.isfinite(v) else float(np.clip(v, 0.0, 1.0))


def regions_to_html(regions: list[dict], page_w: int = IMG_W, page_h: int = IMG_H) -> str:
    if not regions:
        return "<html><body></body></html>"
    ranked = sorted(regions, key=lambda r: r["width"] * r["height"], reverse=True)
    lines = [
        "<!DOCTYPE html>",
        f'<html><body style="width:{page_w}px;height:{page_h}px;margin:0;position:relative;">',
    ]
    for i, r in enumerate(ranked):
        tag = _TAG.get(r.get("element_type", "layout_region"), "div")
        d = int(r.get("depth", 0))
        style = (
            f"position:absolute;left:{r['x']}px;top:{r['y']}px;"
            f"width:{r['width']}px;height:{r['height']}px;"
        )
        pad = "&nbsp;" * min(int(r.get("text_length", 0)), 40)
        lines.append(f'{"  " * (d + 1)}<{tag} id="r{i}" data-depth="{d}" style="{style}">{pad}</{tag}>')
    lines.append("</body></html>")
    return "\n".join(lines)


def _assign_hierarchy(regions: list[dict]) -> list[dict]:
    if not regions:
        return regions
    areas = [max(1, r["width"] * r["height"]) for r in regions]
    order = sorted(range(len(regions)), key=lambda i: areas[i], reverse=True)
    parent = [-1] * len(regions)
    kids: list[list[int]] = [[] for _ in regions]

    for i in order:
        x, y, w, h = regions[i]["x"], regions[i]["y"], regions[i]["width"], regions[i]["height"]
        best, best_a = -1, 10**18
        for j in order:
            if j == i or areas[j] <= areas[i]:
                continue
            ox, oy, ow, oh = regions[j]["x"], regions[j]["y"], regions[j]["width"], regions[j]["height"]
            inter = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
            if inter / areas[i] >= 0.7 and areas[j] < best_a:
                best, best_a = j, areas[j]
        parent[i] = best
        if best >= 0:
            kids[best].append(i)

    def depth(i: int, seen: set[int] | None = None) -> int:
        seen = seen or set()
        if i in seen or parent[i] < 0:
            return 0
        seen.add(i)
        return 1 + depth(parent[i], seen)

    page = float(IMG_W * IMG_H)
    out = []
    for i, r in enumerate(regions):
        item = dict(r)
        item["depth"] = float(depth(i))
        item["n_children"] = float(len(kids[i]))
        p = parent[i]
        item["parent_area_ratio"] = areas[i] / max(areas[p], 1) if p >= 0 else areas[i] / page
        out.append(item)
    return out


def _region_vector(image_bgr: np.ndarray, edges: np.ndarray, region: dict) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    x = int(np.clip(region["x"], 0, W - 1))
    y = int(np.clip(region["y"], 0, H - 1))
    w = int(np.clip(region["width"], 1, W - x))
    h = int(np.clip(region["height"], 1, H - y))
    crop = image_bgr[y : y + h, x : x + w]
    gray = crop.mean(axis=2) if crop.size else np.zeros((1, 1), np.float32)
    edge = edges[y : y + h, x : x + w]

    xn, yn, wn, hn = x / W, y / H, w / W, h / H
    area = wn * hn
    vals = [
        xn,
        yn,
        wn,
        hn,
        area,
        area,
        xn + wn / 2,
        yn + hn / 2,
        (w / max(h, 1)) / 8.0,
        gray.mean() / 255.0,
        gray.std() / 128.0,
        edge.mean() / 255.0,
        (gray < 245).mean(),
        np.log1p(max(float(region.get("text_length", 0)), 0)) / np.log1p(400),
        max(float(region.get("font_size", 0)), 0) / 64.0,
        min(float(region.get("depth", 0)), 8) / 8.0,
        min(float(region.get("n_children", 0)), 16) / 16.0,
        float(np.clip(region.get("parent_area_ratio", area), 0, 1)),
    ]
    numeric = np.array([_clip01(float(v)) for v in vals], dtype=np.float32)
    one_hot = np.zeros(len(ELEMENT_TYPES), dtype=np.float32)
    one_hot[_TYPE_INDEX.get(region.get("element_type", "layout_region"), _TYPE_INDEX["layout_region"])] = 1.0
    return np.concatenate([numeric, one_hot])


def pack_regions(image_bgr: np.ndarray, regions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    import cv2

    image_bgr = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    regions = _assign_hierarchy(_dedupe(regions) if regions else [])
    if not regions:
        regions = _assign_hierarchy(detect_regions(image_bgr)) or [
            {
                "x": 0,
                "y": 0,
                "width": IMG_W,
                "height": IMG_H,
                "element_type": "layout_region",
                "text_length": 0.0,
                "font_size": 0.0,
                "depth": 0.0,
                "n_children": 0.0,
                "parent_area_ratio": 1.0,
            }
        ]

    features = np.zeros((MAX_REGIONS, STRUCT_DIM), np.float32)
    mask = np.zeros(MAX_REGIONS, np.float32)
    kept = regions[:MAX_REGIONS]
    for i, region in enumerate(kept):
        features[i] = _region_vector(image_bgr, edges, region)
        mask[i] = 1.0
    return features, mask, kept, regions_to_html(kept)


def image_to_dom(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    import cv2

    image_bgr = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return pack_regions(image_bgr, detect_regions(image_bgr))


def boxes_to_dom(image_bgr: np.ndarray, boxes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    import cv2

    image_bgr = cv2.resize(image_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return pack_regions(image_bgr, normalize_dom_boxes(image_bgr, boxes))


def feature_names() -> list[str]:
    return list(NUMERIC_FEATURES) + [f"element_type_{t}" for t in ELEMENT_TYPES]

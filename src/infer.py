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


def display_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Percentile-stretch a saliency map for visualization only.

    Metrics keep the raw prediction. An initialized decoder can sit near 0.5
    with almost no range; stretching the 2nd–98th percentiles makes the spatial
    variation visible without changing stored scores or checkpoint weights.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    lo = float(np.percentile(h, 2))
    hi = float(np.percentile(h, 98))
    if hi - lo < 1e-6:
        return np.zeros_like(h, dtype=np.float32)
    return np.clip((h - lo) / (hi - lo), 0.0, 1.0)


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    vis = display_heatmap(heatmap)
    return cv2.applyColorMap(np.clip(vis * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

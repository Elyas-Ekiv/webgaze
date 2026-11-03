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
    return model, device, ckpt["model_name"]


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


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(heatmap * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

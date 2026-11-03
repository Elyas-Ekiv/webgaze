"""Evaluate every saved model on held-out FiWi and write the results figures."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import FIGURES, IMG_H, IMG_W, MODELS_DIR, RESULTS, ROOT
from src.data.dataset import AttentionDataset, load_manifest
from src.infer import heatmap_to_color, load_checkpoint, predict_bgr
from src.metrics.saliency import average_scores, score_pair
from src.models.multimodal import build_model


def center_bias() -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, IMG_H)
    xs = np.linspace(-1.0, 1.0, IMG_W)
    xx, yy = np.meshgrid(xs, ys)
    bias = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.35 ** 2))
    return (bias / bias.max()).astype(np.float32)


@torch.no_grad()
def score_model(name: str, frame: pd.DataFrame, device: torch.device) -> dict[str, float]:
    if name == "center_bias":
        bias = center_bias()
        scores = []
        for _, row in frame.iterrows():
            target = cv2.imread(row["saliency"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            fixation = cv2.imread(row["fixation"], cv2.IMREAD_GRAYSCALE)
            scores.append(score_pair(bias, target, fixation > 127))
        return average_scores(scores)

    path = MODELS_DIR / f"{name}.pth"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    loader_rows = frame.reset_index(drop=True)
    scores = []
    dataset = AttentionDataset(loader_rows)
    for index in range(len(dataset)):
        image, struct, mask, target, fixation = dataset[index]
        pred = model(
            image.unsqueeze(0).to(device),
            struct.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )
        scores.append(
            score_pair(
                pred[0, 0].float().cpu().numpy(),
                target[0].numpy(),
                fixation[0].numpy(),
            )
        )
    return average_scores(scores)


def plot_metrics(table: dict[str, dict[str, float]], dest: Path) -> None:
    metrics = ["CC", "NSS", "AUC-Judd", "SIM", "KL-Divergence"]
    models = list(table)
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        values = [table[model][metric] for model in models]
        ax.barh(models, values, color="#0f6e6e")
        ax.set_title(metric)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("FiWi saliency metrics")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plot_history(dest: Path) -> None:
    history_path = RESULTS / "history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for name, group in frame.groupby("model"):
        ax.plot(group["epoch"], group["CC"], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CC")
    ax.set_title("Validation correlation during training")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plot_qualitative(model, device, frame: pd.DataFrame, dest: Path) -> None:
    sample = frame.sample(n=min(6, len(frame)), random_state=42)
    fig, axes = plt.subplots(len(sample), 4, figsize=(12, 2.3 * len(sample)), constrained_layout=True)
    if len(sample) == 1:
        axes = np.array([axes])
    for row_index, (_, row) in enumerate(sample.iterrows()):
        image = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        target = cv2.imread(row["saliency"], cv2.IMREAD_GRAYSCALE)
        heatmap, _ = predict_bgr(model, cv2.imread(row["image"]), device)
        overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.55, heatmap_to_color(heatmap), 0.45, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        titles = ["Webpage", "Human saliency", "Prediction", "Overlay"]
        panels = [image, target, heatmap, overlay]
        cmaps = [None, "jet", "jet", None]
        for col, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            axes[row_index, col].imshow(panel, cmap=cmap)
            axes[row_index, col].set_axis_off()
            if row_index == 0:
                axes[row_index, col].set_title(title)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_manifest()
    fiwi = frame[frame.dataset == "fiwi"]
    if fiwi.empty:
        raise RuntimeError("FiWi is missing from the manifest. It must stay held out for testing.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ["center_bias", "resnet_only", "mlp_only", "transformer_only", "vit_only", "concat", "attention"]
    table = {}
    for name in names:
        print("evaluating", name)
        table[name] = score_model(name, fiwi, device)
        print(name, {key: round(value, 4) for key, value in table[name].items()})

    fusion = {name: table[name]["CC"] for name in ("concat", "attention")}
    best_name = max(fusion, key=fusion.get)
    source = MODELS_DIR / f"{best_name}.pth"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    torch.save(payload, MODELS_DIR / "best_multimodal_model.pth")
    (RESULTS / "metrics.json").write_text(
        json.dumps({"fiwi": table, "best_model": best_name, "fiwi_images": int(len(fiwi))}, indent=2),
        encoding="utf-8",
    )
    plot_metrics(table, FIGURES / "fiwi_metrics.png")
    plot_history(FIGURES / "training_curves.png")
    model, device, _ = load_checkpoint(MODELS_DIR / "best_multimodal_model.pth", device)
    plot_qualitative(model, device, fiwi, FIGURES / "fiwi_qualitative.png")
    example_dir = ROOT / "app" / "static" / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fiwi.iloc[0]["image"], example_dir / "sample.jpg")
    print("best model", best_name)


if __name__ == "__main__":
    main()

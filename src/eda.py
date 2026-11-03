"""Exploratory figures used by the project report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from src.config import DATA_PROC, DATA_RAW, FIGURES
from src.data.dataset import load_manifest


def _sample_grid(frame: pd.DataFrame, dataset: str, column: str, title: str, dest, cmap=None) -> None:
    rows = frame[frame.dataset == dataset]
    if rows.empty:
        return
    chosen = rows.sample(n=min(6, len(rows)), random_state=7)
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.2 * len(chosen), 2.6), constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        if column == "image":
            image = cv2.cvtColor(cv2.imread(row[column]), cv2.COLOR_BGR2RGB)
            ax.imshow(image)
        else:
            image = cv2.imread(row[column], cv2.IMREAD_GRAYSCALE)
            ax.imshow(image, cmap=cmap or "jet")
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def _overlay_grid(frame: pd.DataFrame, dataset: str, dest) -> None:
    rows = frame[frame.dataset == dataset]
    if rows.empty:
        return
    chosen = rows.sample(n=min(4, len(rows)), random_state=3)
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.6 * len(chosen), 2.8), constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        image = cv2.imread(row["image"])
        heat = cv2.imread(row["saliency"], cv2.IMREAD_GRAYSCALE)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, color, 0.4, 0)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
    fig.suptitle(f"{dataset} heatmap overlay")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def _fixation_scatter(frame: pd.DataFrame, dest) -> None:
    rows = frame[frame.dataset == "ueyes"]
    if rows.empty:
        return
    xs = []
    ys = []
    for _, row in rows.sample(n=min(40, len(rows)), random_state=1).iterrows():
        fix = cv2.imread(row["fixation"], cv2.IMREAD_GRAYSCALE)
        points = np.column_stack(np.nonzero(fix > 127))
        if len(points) == 0:
            continue
        take = points if len(points) <= 80 else points[:: max(len(points) // 80, 1)]
        ys.extend((take[:, 0] / fix.shape[0]).tolist())
        xs.extend((take[:, 1] / fix.shape[1]).tolist())
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.hist2d(xs, ys, bins=30, cmap="magma")
    ax.set_title("UEyes webpage fixation distribution")
    ax.set_xlabel("Normalized x")
    ax.set_ylabel("Normalized y")
    ax.invert_yaxis()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def _gantt(dest, title: str, tasks: list[tuple[str, int, int]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    for index, (name, start, duration) in enumerate(tasks):
        ax.barh(index, duration, left=start, color="#0f6e6e")
        ax.text(start + duration + 0.1, index, name, va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Month index")
    ax.set_title(title)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_manifest()
    counts = frame.groupby("dataset").size()
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    counts.plot(kind="bar", ax=ax, color="#0f6e6e")
    ax.set_title("Prepared images by dataset")
    ax.set_xlabel("")
    ax.set_ylabel("Images")
    fig.savefig(FIGURES / "dataset_file_counts.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    for dataset, group in frame.groupby("dataset"):
        ax.hist(group["orig_w"], bins=20, alpha=0.55, label=dataset)
    ax.set_title("Image width distribution")
    ax.set_xlabel("Width (pixels)")
    ax.legend()
    fig.savefig(FIGURES / "image_width_distribution.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    for dataset, group in frame.groupby("dataset"):
        ax.hist(group["orig_h"], bins=20, alpha=0.55, label=dataset)
    ax.set_title("Image height distribution")
    ax.set_xlabel("Height (pixels)")
    ax.legend()
    fig.savefig(FIGURES / "image_height_distribution.png", dpi=140)
    plt.close(fig)

    _sample_grid(frame, "salicon", "image", "SALICON training images", FIGURES / "sample_salicon.png")
    _sample_grid(frame, "ueyes", "image", "UEyes webpage screenshots", FIGURES / "sample_ueyes.png")
    _sample_grid(frame, "fiwi", "image", "FiWi webpage stimuli", FIGURES / "sample_fiwi.png")
    _sample_grid(frame, "salicon", "saliency", "SALICON saliency maps", FIGURES / "sample_salicon_maps.png", "jet")
    _sample_grid(frame, "ueyes", "saliency", "UEyes heatmaps", FIGURES / "sample_ueyes_maps.png", "jet")
    _sample_grid(frame, "fiwi", "saliency", "FiWi eye maps", FIGURES / "sample_fiwi_maps.png", "jet")
    _overlay_grid(frame, "ueyes", FIGURES / "sample_ueyes_overlay.png")
    _overlay_grid(frame, "fiwi", FIGURES / "sample_fiwi_overlay.png")
    _fixation_scatter(frame, FIGURES / "ueyes_fixation_distribution.png")
    _gantt(
        FIGURES / "gantt_project1.png",
        "Project 1",
        [
            ("Literature review", 0, 2),
            ("Dataset planning", 1, 2),
            ("Methodology", 2, 2),
            ("Report writing", 3, 2),
        ],
    )
    _gantt(
        FIGURES / "gantt_project2.png",
        "Project 2",
        [
            ("Download and EDA", 0, 1),
            ("Model training", 1, 2),
            ("FiWi evaluation", 2, 1),
            ("Demo and report", 3, 1),
        ],
    )
    raw_note = DATA_RAW.exists()
    print("EDA figures written", FIGURES, "raw present", raw_note)


if __name__ == "__main__":
    main()

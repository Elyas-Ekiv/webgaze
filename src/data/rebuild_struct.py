"""Rebuild HTML/DOM structural features from already-processed images.

Use this after changing STRUCT_DIM so you do not re-download or re-resize maps.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import cv2

from src.config import DATA_PROC
from src.data.html_dom import image_to_dom


def main() -> None:
    manifest = DATA_PROC / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}. Run python -m src.data.preprocess first.")
    frame = pd.read_csv(manifest)
    struct_dir = DATA_PROC / "struct"
    struct_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="rebuild-struct"):
        image = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        features, mask, regions, html = image_to_dom(image)
        stem = row["id"]
        out = struct_dir / f"{stem}.npz"
        html_out = struct_dir / f"{stem}.html"
        import numpy as np

        np.savez_compressed(out, features=features, mask=mask, regions=np.array(regions, dtype=object))
        html_out.write_text(html, encoding="utf-8")
        # Keep manifest struct path pointing at the npz.
        if str(row["struct"]) != str(out):
            frame.loc[frame.id == stem, "struct"] = str(out)
    frame.to_csv(manifest, index=False)
    print("rebuilt", len(frame), "structural feature files at dim", features.shape[-1])


if __name__ == "__main__":
    main()

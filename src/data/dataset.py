"""Torch dataset over the preprocessed manifest."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DATA_PROC


def load_manifest() -> pd.DataFrame:
    path = DATA_PROC / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.data.preprocess first.")
    return pd.read_csv(path)


class AttentionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.rows = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        bgr = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

        sal = cv2.imread(str(row["saliency"]), cv2.IMREAD_GRAYSCALE)
        fix = cv2.imread(str(row["fixation"]), cv2.IMREAD_GRAYSCALE)
        sal_t = torch.from_numpy(sal.astype(np.float32) / 255.0).unsqueeze(0)
        fix_t = torch.from_numpy((fix > 127).astype(np.float32)).unsqueeze(0)

        packed = np.load(row["struct"])
        struct = torch.from_numpy(packed["features"]).float()
        mask = torch.from_numpy(packed["mask"]).float()
        return image, struct, mask, sal_t, fix_t

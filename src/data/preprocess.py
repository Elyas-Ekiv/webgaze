"""Resize screenshots, build layout features, and write the training manifest."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.config import DATA_PROC, DATA_RAW, IMG_H, IMG_W
from src.data.html_dom import image_to_dom

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAP_EXT = IMAGE_EXT | {".mat"}


def _files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MAP_EXT and not path.name.startswith(".")]


def _read_color(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _read_array(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "gaze" in payload:
            return _rasterize_gaze(payload)
        best = None
        for key, value in payload.items():
            if key.startswith("_") or not isinstance(value, np.ndarray):
                continue
            if value.ndim == 2 and value.size > best_size(best):
                best = value
            elif value.ndim == 3 and value.shape[-1] in {1, 3} and value.size > best_size(best):
                best = value[..., 0]
        if best is None:
            return None
        return np.asarray(best, dtype=np.float32)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 0]
    return image.astype(np.float32)


def best_size(array: np.ndarray | None) -> int:
    return 0 if array is None else int(array.size)


def _rasterize_gaze(payload: dict) -> np.ndarray | None:
    resolution = np.array(payload["resolution"]).ravel()
    if resolution.size < 2:
        return None
    height, width = int(resolution[0]), int(resolution[1])
    if height < 2 or width < 2:
        return None
    canvas = np.zeros((height, width), dtype=np.float32)
    for subject in np.array(payload["gaze"]).ravel():
        points = np.array(getattr(subject, "fixations", []))
        if points.size == 0:
            continue
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[-1] < 2:
            continue
        xs = points[:, 0].astype(int)
        ys = points[:, 1].astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        canvas[ys[valid], xs[valid]] = 1.0
    return canvas


def _unit(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0)
    peak = float(array.max()) if array.size else 0.0
    if peak > 1.0:
        array = array / 255.0 if peak > 1.5 else array / peak
    return np.clip(array, 0.0, 1.0)


def _is_sparse(array: np.ndarray) -> bool:
    return int(np.unique(np.round(array, 3)).size) <= 12


def _prepare_maps(saliency: np.ndarray, fixation: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    saliency = _unit(saliency)
    sparse = _is_sparse(saliency)
    if fixation is not None:
        fix_source = (_unit(fixation) > 0.2).astype(np.float32)
    elif sparse:
        fix_source = (saliency > 0.2).astype(np.float32)
    else:
        fix_source = None

    if sparse:
        base = fix_source if fix_source is not None else (saliency > 0.2).astype(np.float32)
        fix_bin = cv2.resize(base, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        blurred = gaussian_filter(fix_bin, sigma=8)
        peak = float(blurred.max())
        saliency_out = blurred / peak if peak > 0 else blurred
        return saliency_out.astype(np.float32), (fix_bin > 0.2).astype(np.float32)

    saliency_out = cv2.resize(saliency, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    if fix_source is None:
        if float(saliency_out.max()) > 0:
            fix_bin = (saliency_out >= np.quantile(saliency_out, 0.9)).astype(np.float32)
        else:
            fix_bin = np.zeros_like(saliency_out)
    else:
        fix_bin = cv2.resize(fix_source, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        fix_bin = (fix_bin > 0.2).astype(np.float32)
    return saliency_out.astype(np.float32), fix_bin.astype(np.float32)


def _index(paths: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in paths:
        found.setdefault(path.stem.lower(), path)
    return found


def _choose_dir(root: Path, names: list[str]) -> Path | None:
    if not root.exists():
        return None
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in names and path.name.lower() not in found:
            found[path.name.lower()] = path
    for name in names:
        if name in found:
            return found[name]
    return None


def _pair_dirs(stimuli: Path, saliency: Path, fixation: Path | None, dataset: str, ui_type: str) -> list[dict]:
    stim_files = [path for path in _files(stimuli) if path.suffix.lower() in IMAGE_EXT]
    sal_index = _index(_files(saliency))
    fix_index = _index(_files(fixation)) if fixation else {}
    chosen: dict[str, Path] = {}
    for image_path in stim_files:
        stem = image_path.stem.lower()
        current = chosen.get(stem)
        if current is None or len(image_path.parts) > len(current.parts):
            chosen[stem] = image_path
    rows = []
    for stem, image_path in chosen.items():
        sal_path = sal_index.get(stem)
        if sal_path is None or sal_path.resolve() == image_path.resolve():
            continue
        split = "train"
        parts = {part.lower() for part in image_path.parts}
        if "val" in parts or "valid" in parts or "validation" in parts:
            split = "val"
        rows.append(
            {
                "dataset": dataset,
                "split_hint": split,
                "ui_type": ui_type,
                "image_path": image_path,
                "saliency_path": sal_path,
                "fixation_path": fix_index.get(stem),
            }
        )
    return rows


def _ueyes_categories(root: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for csv_path in root.rglob("*.csv"):
        try:
            frame = pd.read_csv(csv_path, sep=None, engine="python", nrows=5000)
        except Exception:
            continue
        if frame.shape[1] < 2:
            continue
        columns = {col.lower().strip().replace(" ", "_"): col for col in frame.columns}
        name_col = next((columns[key] for key in ("filename", "file_name", "image_name", "img_name", "image", "img", "media_name", "name", "stimulus") if key in columns), None)
        type_col = next((columns[key] for key in ("category", "ui_type", "interface_type", "type", "interface", "label") if key in columns), None)
        if name_col is None or type_col is None:
            continue
        for _, row in frame.iterrows():
            stem = Path(str(row[name_col])).stem.lower()
            labels[stem] = str(row[type_col]).strip().lower()
        if labels:
            return labels
    return labels


def discover_records() -> list[dict]:
    records: list[dict] = []
    salicon = DATA_RAW / "salicon"
    if (salicon / "stimuli").exists() and (salicon / "saliency").exists():
        records.extend(
            _pair_dirs(
                salicon / "stimuli",
                salicon / "saliency",
                salicon / "fixations" if (salicon / "fixations").exists() else None,
                "salicon",
                "natural",
            )
        )

    ueyes_root = DATA_RAW / "ueyes"
    image_dir = _choose_dir(ueyes_root, ["images", "screenshots", "stimuli"])
    heat_dir = _choose_dir(ueyes_root, ["heatmaps_3s", "heatmaps", "saliency_maps"])
    fix_dir = _choose_dir(ueyes_root, ["fixmaps_3s", "fixmaps", "fixations"])
    if image_dir and heat_dir:
        labels = _ueyes_categories(ueyes_root)
        paired = _pair_dirs(image_dir, heat_dir, fix_dir, "ueyes", "unknown")
        for row in paired:
            label = labels.get(Path(row["image_path"]).stem.lower(), "")
            path_text = str(row["image_path"]).lower()
            kind = label or path_text
            if "webpage" in kind or ("web" in kind and "mobile" not in kind):
                row["ui_type"] = "webpage"
            elif "desktop" in kind:
                row["ui_type"] = "desktop"
            elif "mobile" in kind or "phone" in kind:
                row["ui_type"] = "mobile"
            elif "poster" in kind:
                row["ui_type"] = "poster"
            else:
                row["ui_type"] = "unknown"
            if row["ui_type"] == "webpage":
                records.append(row)

    fiwi_root = DATA_RAW / "fiwi"
    stim = _choose_dir(fiwi_root, ["stimuli", "images", "screenshot", "screenshots"])
    if stim is None:
        pngs = [path for path in _files(fiwi_root) if "stimul" in str(path).lower() or "image" in str(path.parent).lower()]
        if pngs:
            stim = pngs[0].parent
    sal = None
    for path in _files(fiwi_root):
        text = str(path).lower()
        if "all5" in text or "eyemap" in text or "saliency" in text or "heatmap" in text:
            sal = path.parent
            break
    if stim and sal:
        paired = _pair_dirs(stim, sal, None, "fiwi", "webpage")
        for row in paired:
            row["split_hint"] = "test"
            records.append(row)
    elif stim:
        # Some FiWi releases keep fixation maps beside stimuli with a shared stem.
        paired = _pair_dirs(stim, fiwi_root, None, "fiwi", "webpage")
        for row in paired:
            if Path(row["image_path"]) == Path(row["saliency_path"]):
                continue
            row["split_hint"] = "test"
            records.append(row)
    return records


def _assign_ueyes_splits(records: list[dict]) -> None:
    """70% / 15% / 15% train/val/test split (sklearn), as in the report."""
    from sklearn.model_selection import train_test_split

    webpage = [row for row in records if row["dataset"] == "ueyes"]
    if not webpage:
        return
    if len(webpage) < 5:
        for row in webpage:
            row["split_hint"] = "train"
        return
    train_rows, hold_rows = train_test_split(webpage, test_size=0.30, random_state=42)
    val_rows, test_rows = train_test_split(hold_rows, test_size=0.50, random_state=42)
    for row in train_rows:
        row["split_hint"] = "train"
    for row in val_rows:
        row["split_hint"] = "val"
    for row in test_rows:
        row["split_hint"] = "test"


def process() -> pd.DataFrame:
    records = discover_records()
    if not records:
        raise RuntimeError("No dataset pairs found. Download the datasets first.")
    _assign_ueyes_splits(records)
    if DATA_PROC.exists():
        shutil.rmtree(DATA_PROC)
    image_dir = DATA_PROC / "images"
    map_dir = DATA_PROC / "maps"
    fix_dir = DATA_PROC / "fixations"
    struct_dir = DATA_PROC / "struct"
    for folder in (image_dir, map_dir, fix_dir, struct_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    seen: dict[str, int] = {}
    for record in tqdm(records, desc="preprocess"):
        color = _read_color(Path(record["image_path"]))
        raw_map = _read_array(Path(record["saliency_path"]))
        if color is None or raw_map is None:
            continue
        raw_fix = _read_array(Path(record["fixation_path"])) if record.get("fixation_path") else None
        orig_h, orig_w = color.shape[:2]
        resized = cv2.resize(color, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        saliency, fixation = _prepare_maps(raw_map, raw_fix)
        features, mask, regions, html = image_to_dom(resized)

        stem = Path(record["image_path"]).stem.lower()
        key = f"{record['dataset']}_{stem}"
        seen[key] = seen.get(key, 0) + 1
        sample_id = key if seen[key] == 1 else f"{key}_{seen[key]}"

        image_out = image_dir / f"{sample_id}.jpg"
        sal_out = map_dir / f"{sample_id}.png"
        fix_out = fix_dir / f"{sample_id}.png"
        struct_out = struct_dir / f"{sample_id}.npz"
        html_out = struct_dir / f"{sample_id}.html"
        cv2.imwrite(str(image_out), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(sal_out), np.clip(saliency * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(fix_out), np.clip(fixation * 255.0, 0, 255).astype(np.uint8))
        np.savez_compressed(struct_out, features=features, mask=mask, regions=np.array(regions, dtype=object))
        html_out.write_text(html, encoding="utf-8")

        rows.append(
            {
                "id": sample_id,
                "dataset": record["dataset"],
                "split": record["split_hint"],
                "ui_type": record["ui_type"],
                "orig_w": orig_w,
                "orig_h": orig_h,
                "image": str(image_out),
                "saliency": str(sal_out),
                "fixation": str(fix_out),
                "struct": str(struct_out),
            }
        )

    frame = pd.DataFrame(rows)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PROC / "manifest.csv", index=False)
    with (DATA_PROC / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "count"])
        for (dataset, split), group in frame.groupby(["dataset", "split"]):
            writer.writerow([dataset, split, len(group)])
    print(frame.groupby(["dataset", "split"]).size())
    return frame


def main() -> None:
    process()


if __name__ == "__main__":
    main()

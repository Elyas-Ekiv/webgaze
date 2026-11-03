"""SALICON pretrain → UEyes fine-tune. FiWi picks the winner in evaluate."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, RESULTS
from src.data.dataset import AttentionDataset, load_manifest
from src.metrics.saliency import average_scores, score_pair
from src.models.multimodal import build_model, saliency_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        AttentionDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        image, struct, mask, target, fixation = [item.to(device, non_blocking=True) for item in batch]
        with torch.set_grad_enabled(train):
            pred = model(image, struct, mask)
            loss = saliency_loss(pred.float(), target, fixation)
        if train:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * image.size(0)
        count += image.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    count = 0
    scores = []
    for batch in loader:
        image, struct, mask, target, fixation = [item.to(device, non_blocking=True) for item in batch]
        pred = model(image, struct, mask)
        loss = saliency_loss(pred.float(), target, fixation)
        total += float(loss.detach()) * image.size(0)
        count += image.size(0)
        pred_np = pred.float().cpu().numpy()
        target_np = target.cpu().numpy()
        fix_np = fixation.cpu().numpy()
        for index in range(pred_np.shape[0]):
            scores.append(score_pair(pred_np[index, 0], target_np[index, 0], fix_np[index, 0]))
    return total / max(count, 1), average_scores(scores)


def fit(model, train_loader, val_loader, device, epochs: int, lr: float, tag: str, history: list) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_cc = -1e9
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        val_loss, scores = evaluate_loader(model, val_loader, device)
        row = {
            "model": tag,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **scores,
        }
        history.append(row)
        print(
            f"{tag} epoch {epoch}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"CC {scores['CC']:.4f}  NSS {scores['NSS']:.4f}  KL {scores['KL-Divergence']:.4f}"
        )
        if scores["CC"] > best_cc:
            best_cc = scores["CC"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best_cc)


def copy_matching(destination, source_state: dict) -> None:
    own = destination.state_dict()
    transferred = 0
    for key, value in source_state.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            transferred += 1
    destination.load_state_dict(own)
    print(f"transferred {transferred} tensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-pretrain", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="resnet_only,mlp_only,transformer_only,vit_only,concat,attention",
        help="Comma-separated fine-tune model names",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    frame = load_manifest()
    sal_train = frame[(frame.dataset == "salicon") & (frame.split == "train")]
    sal_val = frame[(frame.dataset == "salicon") & (frame.split == "val")]
    if sal_val.empty and len(sal_train) > 20:
        sal_val = sal_train.sample(min(400, max(1, len(sal_train) // 10)), random_state=args.seed)
        sal_train = sal_train.drop(sal_val.index)
    elif len(sal_val) > 400:
        sal_val = sal_val.sample(400, random_state=args.seed)
    web_train = frame[(frame.dataset == "ueyes") & (frame.split == "train")]
    web_val = frame[(frame.dataset == "ueyes") & (frame.split == "val")]
    if web_train.empty or web_val.empty:
        raise RuntimeError("UEyes webpage train/val split is empty. Check preprocessing labels.")
    if sal_train.empty:
        raise RuntimeError("SALICON training split is empty.")
    print(f"SALICON train {len(sal_train)} val {len(sal_val)} | UEyes train {len(web_train)} val {len(web_val)}")

    # Quick shape check for DOM feature dim.
    sample = AttentionDataset(web_train.iloc[:1])
    _image, struct, _mask, _sal, _fix = sample[0]
    expected = build_model("attention").struct_mlp[0].in_features
    if struct.shape[-1] != expected:
        raise RuntimeError(
            f"Structural feature dim {struct.shape[-1]} != model dim {expected}. "
            "Re-run: python -m src.data.preprocess"
        )

    train_sal = make_loader(sal_train, args.batch_size, True)
    val_sal = make_loader(sal_val, args.batch_size, False)
    train_web = make_loader(web_train, args.batch_size, True)
    val_web = make_loader(web_val, args.batch_size, False)

    history: list[dict] = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    visual = build_model("resnet_only").to(device)
    print("pretraining ResNet+U-Net on SALICON")
    fit(visual, train_sal, val_sal, device, args.epochs_pretrain, 1e-4, "resnet_pretrain", history)
    visual_state = {key: value.detach().cpu().clone() for key, value in visual.state_dict().items()}
    torch.save({"model_name": "resnet_only", "state_dict": visual_state}, MODELS_DIR / "resnet_pretrain.pth")

    finetune_names = [name.strip() for name in args.models.split(",") if name.strip()]
    val_scores: dict[str, float] = {}
    for name in finetune_names:
        print(f"fine-tuning {name}")
        model = build_model(name).to(device)
        if name in {"resnet_only", "concat", "attention"}:
            copy_matching(model, visual_state)
        epochs = 4 if name == "vit_only" else args.epochs_finetune
        best_cc = fit(model, train_web, val_web, device, epochs, 1e-4, name, history)
        val_scores[name] = best_cc
        torch.save({"model_name": name, "state_dict": model.state_dict()}, MODELS_DIR / f"{name}.pth")

    # Provisional best multimodal by UEyes validation CC (FiWi decides final in evaluate).
    fusion = {name: val_scores[name] for name in ("concat", "attention") if name in val_scores}
    if fusion:
        provisional = max(fusion, key=fusion.get)
        shutil.copy(MODELS_DIR / f"{provisional}.pth", MODELS_DIR / "best_multimodal_model.pth")
        print("provisional best multimodal", provisional, "UEyes val CC", fusion[provisional])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("saved checkpoints and history")


if __name__ == "__main__":
    main()

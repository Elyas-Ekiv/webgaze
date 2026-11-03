"""Saliency models: ResNet/ViT visual, DOM structural, concat/attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.config import IMG_H, IMG_W, IMAGENET_MEAN, IMAGENET_STD, STRUCT_DIM

NAMES = {"resnet_only", "vit_only", "mlp_only", "transformer_only", "concat", "attention"}


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _cat_skip(x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
    if skip is None:
        return torch.cat([x, torch.zeros_like(x)], dim=1)
    if x.shape[-2:] != skip.shape[-2:]:
        skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([x, skip], dim=1)


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up3 = _block(512, 128)
        self.up2 = _block(256, 64)
        self.up1 = _block(128, 32)
        self.up0 = _block(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip3: torch.Tensor | None = None,
        skip2: torch.Tensor | None = None,
        skip1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(_cat_skip(x, skip3))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(_cat_skip(x, skip2))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(_cat_skip(x, skip1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.head(self.up0(x))
        x = F.interpolate(x, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, in_ch: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_block(in_ch, 128), _block(128, 64), _block(64, 32), _block(32, 16)])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor, *_, **__) -> torch.Tensor:
        for layer in self.layers:
            x = F.interpolate(layer(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.head(x), size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.proj = nn.Conv2d(512, 256, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {"bottleneck": self.proj(s4), "skip1": s1, "skip2": s2, "skip3": s3}


class StructuralEncoder(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(STRUCT_DIM, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, struct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(struct)
        pad = mask <= 0
        if pad.all(dim=1).any():
            pad = pad.clone()
            pad[pad.all(dim=1), 0] = False
        return self.encoder(tokens, src_key_padding_mask=pad)


class PatchViT(nn.Module):
    def __init__(self, dim: int = 192, depth: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 16, stride=16)
        self.pos = nn.Parameter(torch.zeros(1, 14 * 24, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj = nn.Conv2d(dim, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 8))  # 216 -> 224 so 16px patches land cleanly
        tokens = self.encoder(self.patch(x).flatten(2).transpose(1, 2) + self.pos)
        return self.proj(tokens.transpose(1, 2).reshape(x.shape[0], -1, 14, 24))


class SaliencyModel(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in NAMES:
            raise ValueError(f"Unknown model {name}")
        self.name = name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        uses_resnet = name in {"resnet_only", "concat", "attention"}
        self.decoder = UNetDecoder() if uses_resnet else UpsampleDecoder(256)
        if uses_resnet:
            self.visual = VisualEncoder()
        if name == "vit_only":
            self.vit = PatchViT()
        if name in {"mlp_only", "transformer_only", "concat", "attention"}:
            self.struct_mlp = nn.Sequential(
                nn.Linear(STRUCT_DIM, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        if name in {"transformer_only", "concat", "attention"}:
            self.structural = StructuralEncoder(128)
        if name in {"concat", "attention"}:
            self.fuse = nn.Conv2d(256 + 128, 256, 1)
        if name == "attention":
            self.query = nn.Linear(256, 128)
        if name in {"mlp_only", "transformer_only"}:
            self.render_sigma = nn.Parameter(torch.tensor(0.08))

    def _norm(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def _paint(self, weights: torch.Tensor, struct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n = weights.shape
        ys = torch.linspace(0, 1, 54, device=weights.device)
        xs = torch.linspace(0, 1, 96, device=weights.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = self.render_sigma.clamp(0.03, 0.25)
        cx = struct[:, :, 6].clamp(0, 1).view(b, n, 1, 1)
        cy = struct[:, :, 7].clamp(0, 1).view(b, n, 1, 1)
        blob = torch.exp(-((xx.view(1, 1, 54, 96) - cx) ** 2 + (yy.view(1, 1, 54, 96) - cy) ** 2) / (2 * sigma**2))
        canvas = (blob * (weights * mask).view(b, n, 1, 1)).sum(dim=1, keepdim=True)
        return canvas / canvas.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    def _from_struct(self, struct: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        canvas = F.interpolate(self._paint(weights, struct, mask), size=(14, 24), mode="bilinear", align_corners=False)
        return self.decoder(canvas.repeat(1, 256, 1, 1))

    def _decode(self, feats: dict[str, torch.Tensor], fused: torch.Tensor) -> torch.Tensor:
        return self.decoder(fused, feats["skip3"], feats["skip2"], feats["skip1"])

    def forward(self, image: torch.Tensor, struct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.name == "mlp_only":
            return self._from_struct(struct, mask, torch.sigmoid(self.struct_mlp(struct)).squeeze(-1))

        if self.name == "transformer_only":
            tokens = self.structural(struct, mask)
            weights = torch.sigmoid(self.struct_mlp(struct)).squeeze(-1) * torch.sigmoid(tokens.mean(dim=-1))
            return self._from_struct(struct, mask, weights)

        if self.name == "vit_only":
            return self.decoder(self.vit(self._norm(image)))

        feats = self.visual(self._norm(image))
        visual = feats["bottleneck"]
        if self.name == "resnet_only":
            return self._decode(feats, visual)

        tokens = self.structural(struct, mask)
        if self.name == "concat":
            valid = mask.unsqueeze(-1).clamp_min(0)
            pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            tiled = pooled[:, :, None, None].expand(-1, -1, visual.shape[2], visual.shape[3])
            return self._decode(feats, self.fuse(torch.cat([visual, tiled], dim=1)))

        b, _, h, w = visual.shape
        q = self.query(visual.flatten(2).transpose(1, 2))
        scores = torch.matmul(q, tokens.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        scores = scores.masked_fill(mask.unsqueeze(1) <= 0, -1e4)
        context = torch.matmul(torch.softmax(scores, dim=-1), tokens)
        context = context.transpose(1, 2).reshape(b, 128, h, w)
        return self._decode(feats, self.fuse(torch.cat([visual, context], dim=1)))


def build_model(name: str) -> SaliencyModel:
    return SaliencyModel(name)


def saliency_loss(pred: torch.Tensor, target: torch.Tensor, fixation: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    pred = pred.clamp(eps, 1.0)
    target = target.clamp(0.0, 1.0)

    p = pred.flatten(1)
    q = target.flatten(1)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    kl = (q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())).sum(dim=1).mean()

    pc = pred - pred.mean(dim=(2, 3), keepdim=True)
    qc = target - target.mean(dim=(2, 3), keepdim=True)
    cc = (pc * qc).sum(dim=(2, 3)) / torch.sqrt((pc**2).sum(dim=(2, 3)) * (qc**2).sum(dim=(2, 3))).clamp_min(eps)
    cc = cc.mean()

    z = (pred - pred.mean(dim=(2, 3), keepdim=True)) / pred.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    nss = ((z * fixation).sum(dim=(2, 3)) / fixation.sum(dim=(2, 3)).clamp_min(1.0)).mean()
    return kl + (1.0 - cc) + 0.1 * (-nss)

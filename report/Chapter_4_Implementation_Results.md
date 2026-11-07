# Chapter 4: IMPLEMENTATION & RESULTS

**Project:** WebGaze — Multimodal webpage saliency prediction  
**Course:** CSDS4105 · B-Tech Course Project

> **Template note:** In `DS and AI-CSDS4105 - Final Review Report Template-Research`, Chapter 4 is already **Preprocessing Datasets**. This Implementation & Results content is therefore inserted in the official report as **Chapter 7 (sections 7.1–7.14)** in `report/Final_Review_Report_with_Results.docx`. See `Chapter_Implementation_Template_Validation.md`.

---

## 4.1 Implementation Overview

This chapter describes how the WebGaze system was built and what it achieved. The goal is to predict a **saliency heatmap** (where people are likely to look) for a static desktop webpage screenshot, optionally using live DOM layout when a URL is provided.

The implementation follows a clear pipeline:

1. **Download** SALICON, UEyes, and FiWi datasets  
2. **Preprocess** screenshots and saliency maps to 384×216 and extract layout/DOM region features  
3. **Train** models with SALICON pretraining, then UEyes fine-tuning  
4. **Evaluate** on held-out FiWi  
5. **Deploy** a FastAPI demo that accepts an image upload or a live URL  

The shipped model is the best multimodal fusion variant (**concat**), saved as `models/best_multimodal_model.pth`.

---

## 4.2 Development Environment

| Item | Choice |
| --- | --- |
| Language | Python 3 (Anaconda on Windows for local work) |
| Deep learning | PyTorch + torchvision |
| Web demo | FastAPI + Uvicorn |
| Browser capture | Playwright (Chromium) |
| Image / data tools | OpenCV, NumPy, Pandas, Pillow, SciPy |
| Reporting | Matplotlib, Seaborn, python-docx |
| Deployment | Docker / Dokploy (`Dockerfile`, `docker-compose.yml`) |
| OS (dev) | Windows 10 |
| Demo URL | `http://127.0.0.1:8000` |

Key packages are listed in `requirements.txt`. Torch is installed separately (CUDA wheel locally; CPU wheel in the Docker image).

---

## 4.3 Dataset Implementation

Three datasets are used; WebSaliency was planned earlier but **not used** (no reliable public download).

| Dataset | Role | Notes |
| --- | --- | --- |
| **SALICON** | Pretrain visual encoder | Natural-image saliency; teaches general “what looks salient” |
| **UEyes** | Fine-tune on webpages | Webpage screenshots only; train/val splits |
| **FiWi** | Held-out final test | Never used for training or fine-tuning |

**Preprocessing (`src.data.preprocess`):**

- Resize screenshots and ground-truth saliency maps to **384×216**  
- Extract up to **48** layout regions per page with structural features (position, size, brightness, contrast, hierarchy, element type, …)  
- Write `data/processed/manifest.csv` and per-sample `.npz` feature files  
- Mark FiWi rows as `test` only  

**Feature paths:**

| Mode | Region source |
| --- | --- |
| Training / image upload | Estimated layout from the screenshot (`image_to_dom`) |
| Live URL demo | Real DOM boxes via Playwright (`boxes_to_dom`) |

Both paths use the **same feature schema** so one trained model serves both.

---

## 4.4 Model / Algorithm Implementation

Core code lives in `src/models/multimodal.py`.

**Architecture (multimodal path):**

1. **Visual branch** — ResNet-18 encoder with U-Net skip features  
2. **Structural branch** — region feature embedding + Transformer (or MLP for structure-only baselines)  
3. **Fusion** — concatenate tiled structure with visual maps (`concat`) or cross-attention then concat (`attention`)  
4. **Decoder** — U-Net (or upsample decoder for ViT / structure-only) → saliency heatmap  

**Variants trained:**

| Name | Image | Structure | Fusion |
| --- | --- | --- | --- |
| `resnet_only` | ResNet + U-Net | No | — |
| `vit_only` | Patch ViT | No | — |
| `mlp_only` | No | MLP → Gaussian paint | — |
| `transformer_only` | No | Transformer → paint | — |
| `concat` | Yes | Yes | Tile + concat |
| `attention` | Yes | Yes | Cross-attention + concat |

**Loss:** KL divergence + (1 − CC) + 0.1 × (−NSS), implemented as `saliency_loss`.

---

## 4.5 Model Training and Hyperparameter Configuration

Training script: `src/train.py`.

**Procedure:**

1. Seed RNG (`seed=42` by default)  
2. Pretrain `resnet_only` on SALICON  
3. Fine-tune each variant on UEyes (ResNet-based models load matching pretrained weights)  
4. Save checkpoints under `models/` and training curves in `results/history.json`  
5. Final winner among fusion models is decided later on **FiWi** in `src/evaluate.py`

**Default hyperparameters:**

| Hyperparameter | Default |
| --- | --- |
| Input size | 384 × 216 |
| Max regions | 48 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Pretrain epochs (SALICON) | 2 |
| Fine-tune epochs (UEyes) | 6 (`vit_only`: 4) |
| Gradient clip | max norm 1.0 |
| Seed | 42 |

CLI overrides: `--epochs-pretrain`, `--epochs-finetune`, `--batch-size`, `--seed`, `--models`.

---

## 4.6 System / Application Implementation

The interactive prototype is a FastAPI app under `app/`.

| Component | Role |
| --- | --- |
| `app/main.py` | API: health, predict (upload/URL), sample |
| `app/static/index.html` | Demo UI layout |
| `app/static/app.js` | Client: upload/URL, overlay canvas, hotspot UI |
| `app/static/style.css` | Visual theme |
| `src/infer.py` | Load checkpoint and run prediction |

**Main user flows:**

1. Upload a webpage screenshot → heatmap + overlay + ranked hotspots  
2. Enter a URL → Playwright captures screenshot + DOM → same prediction path  
3. Try sample image from FiWi (`/api/sample`)

Docker deployment exposes port **8000** and serves `uvicorn app.main:app`.

---

## 4.7 Experimental Setup

| Aspect | Setting |
| --- | --- |
| Pretrain data | SALICON train (+ small val sample) |
| Fine-tune data | UEyes webpage train/val |
| Test data | FiWi only (**149** images) |
| Baseline | Center-bias Gaussian map |
| Selection rule | Best multimodal fusion by FiWi **CC** → `best_multimodal_model.pth` |
| Metrics | CC ↑, NSS ↑, AUC-Judd ↑, SIM ↑, KL-Divergence ↓ |
| Scope | Static **desktop** pages |

Evaluation script: `src/evaluate.py` → `results/metrics.json`, figures, and RESULTS notes.

---

## 4.8 Model Results

On held-out **FiWi** (149 images), the best model is **`concat`**.

| Model | CC | NSS | AUC-Judd | SIM | KL ↓ |
| --- | --- | --- | --- | --- | --- |
| center_bias | 0.182 | 0.595 | 0.673 | 0.242 | 2.165 |
| resnet_only | 0.279 | 0.960 | 0.761 | 0.287 | 1.772 |
| mlp_only | 0.153 | 0.501 | 0.630 | 0.199 | 2.000 |
| transformer_only | 0.157 | 0.514 | 0.637 | 0.209 | 1.970 |
| vit_only | 0.165 | 0.540 | 0.654 | 0.201 | 1.993 |
| **concat (best)** | **0.315** | **1.090** | **0.775** | **0.297** | **1.658** |
| attention | 0.304 | 1.055 | 0.760 | 0.296 | 1.761 |

Saved artifact: `models/best_multimodal_model.pth` (`best_model = concat`).

---

## 4.9 Performance Evaluation

Saliency is not classification, so there is **no single accuracy %**. Performance is judged with standard map/fixation metrics:

| Metric | Direction | Meaning |
| --- | --- | --- |
| **CC** | Higher better | Correlation with human saliency map |
| **NSS** | Higher better | Predicted values at real fixation locations |
| **AUC-Judd** | Higher better | Ranking fixated vs non-fixated pixels (~0.5 ≈ chance) |
| **SIM** | Higher better | Distribution / histogram overlap |
| **KL-Divergence** | Lower better | How far prediction is from the human map |

**Best reported scores (concat):** CC ≈ **0.315**, NSS ≈ **1.090**, AUC-Judd ≈ **0.775**, SIM ≈ **0.297**, KL ≈ **1.658**.

---

## 4.10 Comparative Results

Key comparisons on FiWi:

- **Multimodal vs visual-only:** `concat` (CC 0.315) beats `resnet_only` (CC 0.279) → layout features help.  
- **Fusion styles:** `concat` slightly ahead of `attention` (CC 0.304).  
- **Structure-only:** `mlp_only` / `transformer_only` underperform → image appearance remains essential.  
- **ViT baseline:** `vit_only` (CC 0.165) is weaker than ResNet under this data/compute setup.  
- **Center bias:** All strong models beat the Gaussian center prior (CC 0.182).

**Defendable claim:** under this SALICON → UEyes → FiWi setup, multimodal fusion outperforms visual-only ResNet on held-out webpage saliency.

---

## 4.11 Error / Failure Analysis

Observed / expected failure modes:

1. **Unusual layouts** — pages with atypical structure or sparse regions can produce weak or diffuse maps.  
2. **Screenshot-only features** — training uses estimated regions; text length / font size are often missing compared with live DOM.  
3. **Structure-only models** — without pixels, performance collapses toward weak priors.  
4. **Domain gap** — SALICON is natural images; UEyes adapts to web UI, but FiWi pages can still differ in style/density.  
5. **Scope limits** — static desktop only; no mobile layouts, scrolling video, or interactive hover states.  
6. **Hard saliency cases** — low-contrast content, ads/clutter, or multiple competing foci reduce agreement with human maps (higher KL, lower CC/SIM).

These limitations motivate reporting multiple metrics rather than a single score, and keeping FiWi strictly held out.

---

## 4.12 System / Prototype Results

The working prototype demonstrates end-to-end use:

- Loads `best_multimodal_model.pth` once (lazy cache)  
- Accepts **file upload** or **URL**  
- Returns screenshot, color heatmap, overlay, region list, and ranked **hotspots**  
- Sample FiWi page available for quick demo  

Operational checks: `GET /api/health`, `POST /api/predict`, `GET /api/sample`. Local run uses Uvicorn on port 8000; containerized run matches Dokploy/Docker deployment.

---

## 4.13 Results Summary

| Item | Outcome |
| --- | --- |
| Best model | `concat` multimodal fusion |
| Test set | FiWi, 149 images (held out) |
| Best CC / AUC-Judd | 0.315 / 0.775 |
| vs ResNet-only | Clear gain from structural branch |
| vs attention fusion | Concat slightly better on FiWi CC |
| Deliverable | Trained weights + FastAPI WebGaze demo |
| Data note | WebSaliency unused; SALICON + UEyes + FiWi only |

Overall, the implementation meets the project aim: a multimodal webpage saliency system with measurable FiWi results and a usable interactive prototype.

---

## 4.14 Chapter Summary

This chapter covered implementation and results for WebGaze. Datasets were prepared and split so FiWi remains an honest final test. Models were implemented with ResNet (and baselines), structural encoders, and concat/attention fusion, then trained with SALICON pretraining and UEyes fine-tuning. The FastAPI application exposes the best checkpoint for screenshot and live-URL demos. On FiWi, **concat** was the strongest model across the reported saliency metrics and outperformed visual-only ResNet, supporting the multimodal design choice while acknowledging layout estimation gaps and static-desktop scope limits.

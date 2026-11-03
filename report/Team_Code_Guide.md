# Webpage Attention — Full Team Code Guide

**Course:** CSDS4105 · B-Tech Course Project  
**Project:** Multimodal AI for predicting human visual attention on webpages  
**Use this for:** reading together before evaluation, sharing with classmates, answering “what does this file do?”

---

## 1. Project in one page

### What we built
A system that takes a **webpage screenshot** (and optionally **live DOM layout** from a URL) and outputs a **saliency heatmap** — a map of where people are likely to look.

### Core idea
Webpages are not natural photos. Layout matters (header, text blocks, buttons). So we use **two inputs**:

1. **Visual branch** — ResNet-18 reads the screenshot  
2. **Structural branch** — Transformer/MLP reads layout or DOM region features  
3. **Fusion** — concat or attention combines both  
4. **Decoder** — U-Net draws the heatmap  

### Training / test story
| Stage | Dataset | Why |
| --- | --- | --- |
| Pretrain | SALICON | Learn general “what looks salient” from many natural images |
| Fine-tune | UEyes (webpage only) | Adapt to webpage / UI layouts |
| Final test | FiWi (held out) | Fair webpage evaluation — never used in training |

### Important facts for viva
- Output is a **heatmap**, not a class label → no single “accuracy %”  
- Metrics: **CC, NSS, AUC-Judd, SIM, KL-Divergence**  
- Best model in our last run: **`concat`** (`models/best_multimodal_model.pth`)  
- WebSaliency was planned earlier but **not used** (no reliable download)  
- Scope: **static desktop** pages only  

---

## 2. End-to-end workflow

### 2.1 Big picture (data → model → demo)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  download   │ ──► │  preprocess  │ ──► │    eda      │ ──► │  train   │
│  raw zips   │     │  384×216 +   │     │  charts     │     │ SALICON→ │
│             │     │  DOM feats   │     │             │     │ UEyes    │
└─────────────┘     └──────────────┘     └─────────────┘     └────┬─────┘
                                                                  │
                                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│    demo     │ ◄── │   infer.py   │ ◄── │  evaluate   │ ◄── │ *.pth    │
│ FastAPI UI  │     │ load best    │     │ FiWi scores │     │checkpoints│
└─────────────┘     └──────────────┘     └──────┬──────┘     └──────────┘
                                                │
                                                ▼
                                         report.py → Word report
```

### 2.2 Commands (run in order)

```powershell
$py = "$env:USERPROFILE\anaconda3\python.exe"
cd "C:\Users\Admin\Desktop\course project"

& $py -m src.data.download      # 1. get datasets into data/raw
& $py -m src.data.preprocess    # 2. resize + DOM features + manifest
& $py -m src.eda                # 3. make EDA figures
& $py -m src.train              # 4. SALICON pretrain + UEyes fine-tune
& $py -m src.evaluate           # 5. FiWi metrics + best_multimodal_model.pth
& $py -m src.report             # 6. append Chapter 7 into Word report
& $py -m playwright install chromium   # once, for URL demo
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

### 2.3 What happens inside training (`src.train`)

```
1. Load manifest.csv
2. Split rows:
     - SALICON train/val
     - UEyes webpage train/val
3. Build resnet_only → train on SALICON (pretrain)
4. Save visual weights
5. For each of: resnet_only, mlp_only, transformer_only, vit_only, concat, attention
     - create model
     - if ResNet-based: copy matching pretrained weights
     - fine-tune on UEyes
     - save models/<name>.pth
6. Write results/history.json (val CC per epoch)
7. Copy provisional best fusion to best_multimodal_model.pth
   (final winner is decided again on FiWi in evaluate)
```

### 2.4 What happens inside evaluation (`src.evaluate`)

```
1. Load FiWi rows only (held out)
2. Score: center_bias, resnet_only, mlp_only, transformer_only,
          vit_only, concat, attention
3. Pick best of concat vs attention by FiWi CC
4. Save that checkpoint as models/best_multimodal_model.pth
5. Write results/metrics.json + RESULTS.md
6. Save figures: fiwi_metrics.png, training_curves.png, fiwi_qualitative.png
```

### 2.5 What happens when you use the website

**Upload screenshot**
```
browser → POST /api/predict (file)
       → OpenCV decode image
       → image_to_dom() estimates layout boxes from pixels
       → ResNet + structure → heatmap
       → JSON with screenshot, heatmap, regions, html dump
```

**Paste URL**
```
browser → POST /api/predict (url)
       → Playwright opens Chromium
       → screenshot + real DOM boxes (getBoundingClientRect)
       → boxes_to_dom() packs same feature schema
       → same model forward
       → JSON response
```

---

## 3. Folder structure

```
course project/
├── app/                         Live website
│   ├── main.py                  FastAPI routes + Playwright capture
│   └── static/
│       ├── index.html           UI page
│       ├── style.css            Styling
│       ├── app.js               Frontend logic (predict, draw canvas)
│       └── examples/sample.jpg  Sample image for “Sample” button
│
├── src/                         All science / training code
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── eda.py
│   ├── report.py
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── rebuild_struct.py
│   │   ├── structural.py
│   │   ├── html_dom.py
│   │   └── dataset.py
│   ├── models/
│   │   └── multimodal.py
│   └── metrics/
│       └── saliency.py
│
├── data/
│   ├── raw/                     Original downloaded datasets
│   └── processed/               Training-ready files
│       ├── images/
│       ├── maps/
│       ├── fixations/
│       ├── struct/              *.npz features + *.html dumps
│       ├── manifest.csv
│       └── summary.csv
│
├── models/                      Trained weights (*.pth)
├── results/                     metrics, figures, RESULTS.md
├── report/                      Word reports + this guide
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 4. File-by-file reference

### 4.1 Root / ops

| File | What it does |
| --- | --- |
| `README.md` | Short setup + run instructions |
| `requirements.txt` | Python deps (numpy, opencv, fastapi, playwright, bs4, …). Torch is installed separately for CUDA/CPU |
| `Dockerfile` | Builds a CPU demo image: installs torch CPU, deps, Chromium, copies `src/`, `app/`, `models/`, runs uvicorn on port 8000 |
| `docker-compose.yml` | One service `attention`, maps host 8000→container 8000, mounts `./models` read-only |

---

### 4.2 `src/config.py`

**Role:** Single place for shared constants.

Contains:
- Paths: `ROOT`, `DATA_RAW`, `DATA_PROC`, `MODELS_DIR`, `RESULTS`, `FIGURES`, `REPORT_DIR`
- Image size: `IMG_W=384`, `IMG_H=216`
- `MAX_REGIONS=48` (max layout boxes per page)
- `NUMERIC_FEATURES` + `ELEMENT_TYPES` → `STRUCT_DIM`
- ImageNet mean/std for ResNet normalization
- Download checksums / Drive IDs for datasets

**If asked “why 384×216?”** — keeps a 16:9 webpage-like ratio and a fixed tensor size for the network.

---

### 4.3 Data layer (`src/data/`)

#### `download.py`
Downloads SALICON, UEyes, FiWi into `data/raw/`.  
Uses curl / parallel ranged downloads / gdown for Google Drive SALICON parts.  
Checks MD5 / file sizes when configured.

**Run:** `python -m src.data.download`

#### `preprocess.py`
Turns raw datasets into a clean training table.

Does:
1. Discover image↔saliency pairs for SALICON, UEyes, FiWi  
2. Keep UEyes **webpage** samples only (filter by labels/paths)  
3. Assign UEyes splits ~70/15/15 with sklearn  
4. Mark FiWi as `test`  
5. Resize images/maps to 384×216  
6. Build fixation maps  
7. Call `image_to_dom()` for structural features  
8. Write `manifest.csv` + `summary.csv`

**Run:** `python -m src.data.preprocess`  
**Warning:** wipes and rebuilds `data/processed/`.

#### `rebuild_struct.py`
Faster path: recompute only DOM/struct features from existing processed images (after feature-schema changes). Does **not** re-download or re-resize saliency maps.

**Run:** `python -m src.data.rebuild_struct`

#### `structural.py`
Computer-vision layout detector / normalizer.

- `detect_regions(image)` — OpenCV edges + panels → candidate boxes, classify as header / text / layout / background  
- `normalize_dom_boxes(image, boxes)` — clean Playwright DOM boxes onto 384×216 canvas  
- `extract_prototype` / `features_from_boxes` — thin wrappers that call `html_dom` packing (kept for older call sites)

This file finds **boxes**. It does not train the neural net.

#### `html_dom.py`
Converts boxes → model input.

- Assigns **hierarchy** (which region nests inside which → depth, n_children, parent_area_ratio)  
- Builds feature vector per region (position, size, brightness, contrast, text_length, font_size, one-hot type, …)  
- Pads/truncates to `MAX_REGIONS`  
- Writes a small **HTML dump** string (for inspection / demo JSON)  
- `image_to_dom(img)` — screenshot path  
- `boxes_to_dom(img, boxes)` — live DOM path  

#### `dataset.py`
PyTorch `Dataset`.

- `load_manifest()` reads `manifest.csv`  
- `__getitem__` returns:  
  `(image_tensor, struct_features, mask, saliency_target, fixation_map)`  

Used by `train.py` and `evaluate.py`.

---

### 4.4 Models (`src/models/multimodal.py`)

**Role:** All neural network variants in one file.

| Class / function | Job |
| --- | --- |
| `VisualEncoder` | ResNet-18 backbone; returns bottleneck + skip maps for U-Net |
| `StructuralEncoder` | Embeds region features; Transformer over region tokens |
| `PatchViT` | Small Vision Transformer baseline (visual-only) |
| `UNetDecoder` | Upsamples with skip connections → heatmap |
| `UpsampleDecoder` | Plain upsample decoder (ViT / structure-only) |
| `SaliencyModel` | Wires the right pieces for a named variant |
| `build_model(name)` | Factory |
| `saliency_loss(...)` | KL + (1−CC) + 0.1×(−NSS) |

**Model names:**

| Name | Uses image? | Uses structure? | Fusion |
| --- | --- | --- | --- |
| `resnet_only` | yes (ResNet+U-Net) | no | — |
| `vit_only` | yes (ViT) | no | — |
| `mlp_only` | no | yes (MLP scores → Gaussian paint) | — |
| `transformer_only` | no | yes (Transformer + paint) | — |
| `concat` | yes | yes | tile pooled structure + concat |
| `attention` | yes | yes | cross-attention then concat |

---

### 4.5 Metrics (`src/metrics/saliency.py`)

Implements standard saliency metrics (Bylinskii-style):

| Function | Metric |
| --- | --- |
| `correlation_coefficient` | CC |
| `normalized_scanpath_saliency` | NSS |
| `auc_judd` | AUC-Judd |
| `similarity` | SIM |
| `kl_divergence` | KL |
| `score_pair` | all five for one image |
| `average_scores` | mean over a list of images |

---

### 4.6 Training / eval / infer scripts

#### `src/train.py`
- Seeds RNG  
- Loads SALICON + UEyes loaders  
- Pretrains ResNet on SALICON  
- Fine-tunes all variants on UEyes  
- Saves `models/<name>.pth` and `results/history.json`  
- Optimizer: Adam  

**CLI knobs:** `--epochs-pretrain`, `--epochs-finetune`, `--batch-size`, `--seed`, `--models`

#### `src/evaluate.py`
- Scores every model on FiWi  
- Includes `center_bias` baseline (Gaussian center blob)  
- Writes `metrics.json`, plots, copies best fusion → `best_multimodal_model.pth`  
- Copies one FiWi image to `app/static/examples/sample.jpg`

#### `src/infer.py`
Inference helpers used by demo + qualitative plots:

- `load_checkpoint()` → model, device, name  
- `predict_bgr(model, image, device, boxes=None)` → heatmap + regions  
- `heatmap_to_color()` → JET colormap BGR image  

#### `src/eda.py`
Builds report figures: sample grids, size histograms, fixation scatter, file counts, simple Gantt images → `results/figures/`.

#### `src/report.py`
Copies the course Word template, appends **Chapter 7: Experimental Results** (table + figures) → `report/Final_Review_Report_with_Results.docx`. Also refreshes `results/RESULTS.md`.

---

### 4.7 Demo app (`app/`)

#### `app/main.py`
FastAPI server.

| Route | Behavior |
| --- | --- |
| `GET /` | Serve `index.html` |
| `GET /api/health` | `{ok, checkpoint}` |
| `POST /api/predict` | file and/or URL → heatmap JSON |
| `GET /api/sample` | predict on sample.jpg |

Important internals:
- `get_model()` — lazy-loads best checkpoint once  
- `_capture(url)` — Playwright screenshot + DOM script  
- `_predict(image, boxes)` — runs model + returns base64 images + regions + html  

#### `app/static/index.html`
Layout: upload / URL form + canvas stage + region chips.

#### `app/static/app.js`
- Calls `/api/predict` or `/api/sample`  
- Draws screenshot + heatmap overlay on canvas  
- Opacity slider, region boxes toggle  
- Groups region chips into a 3×3 page grid  

#### `app/static/style.css`
Visual theme (fonts, panels, colors).

---

### 4.8 Data / models / results on disk

| Path | Contents |
| --- | --- |
| `data/raw/` | Original SALICON / UEyes / FiWi downloads |
| `data/processed/manifest.csv` | Master index: id, dataset, split, paths |
| `data/processed/struct/*.npz` | `features` (48×STRUCT_DIM), `mask` (48,), optional regions |
| `models/*.pth` | Per-model checkpoints: `{model_name, state_dict}` |
| `models/best_multimodal_model.pth` | Winner used by the website |
| `results/metrics.json` | Full FiWi table + `best_model` |
| `results/history.json` | Training curves data |
| `results/figures/` | PNGs for report |
| `results/RESULTS.md` | Human-readable metrics table |
| `report/Final_Review_Report_with_Results.docx` | Official report + results chapter |
| `report/Team_Code_Guide.md` / `.docx` | This document |

---

## 5. Structural features (what the second branch sees)

Each region becomes numbers (then one-hot element type):

**Numeric:** x, y, width, height, area, relative_area, center_x, center_y, aspect_ratio, brightness, contrast, edge_density, non_white_ratio, text_length, font_size, depth, n_children, parent_area_ratio  

**Types:** header | background | text_or_content | layout_region  

Up to **48** regions; unused slots are masked out so the Transformer ignores them.

### Screenshot path vs URL path

| | Training / upload | URL demo |
| --- | --- | --- |
| Box source | OpenCV estimates from pixels | Real DOM via Playwright |
| text_length / font_size | Often 0 (no HTML) | From live elements |
| Feature schema | Same | Same |
| Model | Same weights | Same weights |

**Say this clearly in evaluation:** training mostly uses **prototype / estimated layout**, not full page HTML. The URL demo shows the **intended real-DOM path**.

---

## 6. Metrics cheat sheet

| Metric | Want | Meaning |
| --- | --- | --- |
| CC | ↑ | Correlation with human saliency map |
| NSS | ↑ | Predicted values at real fixation points |
| AUC-Judd | ↑ | Ranking fixated vs non-fixated pixels (~0.5 = chance) |
| SIM | ↑ | Histogram / distribution overlap |
| KL | ↓ | How far prediction is from human distribution |

---

## 7. Latest FiWi numbers (149 images)

| Model | CC | NSS | AUC-Judd | SIM | KL |
| --- | --- | --- | --- | --- | --- |
| center_bias | 0.182 | 0.595 | 0.673 | 0.242 | 2.165 |
| resnet_only | 0.279 | 0.960 | 0.761 | 0.287 | 1.772 |
| mlp_only | 0.153 | 0.501 | 0.630 | 0.199 | 2.000 |
| transformer_only | 0.157 | 0.514 | 0.637 | 0.209 | 1.970 |
| vit_only | 0.165 | 0.540 | 0.654 | 0.201 | 1.993 |
| **concat (best)** | **0.315** | **1.090** | **0.775** | **0.297** | **1.658** |
| attention | 0.304 | 1.055 | 0.760 | 0.296 | 1.761 |

**Claim you can defend:** multimodal fusion beats visual-only ResNet on held-out FiWi under this setup.

---

## 8. Common evaluation Q&A

**Q: What is your accuracy?**  
A: Saliency doesn’t use classification accuracy. We report CC / NSS / AUC-Judd / SIM / KL. Best AUC-Judd ≈ **0.775**, CC ≈ **0.315**.

**Q: Why multimodal?**  
A: Webpages have structure. Visual-only models miss layout cues. Our concat model beats `resnet_only`.

**Q: Do you use real HTML in training?**  
A: Feature schema matches DOM fields. Training screenshots use estimated regions (image-to-DOM). Live URL inference uses real DOM boxes from Playwright.

**Q: Why not WebSaliency?**  
A: No reliable public download; report Ch.5 already notes we use SALICON + UEyes + FiWi.

**Q: What is FiWi for?**  
A: Held-out final test only — never fine-tuned on it.

**Q: Which file is the “main model”?**  
A: `src/models/multimodal.py`. Best weights: `models/best_multimodal_model.pth`.

**Q: How do we run the demo?**  
A: `uvicorn app.main:app --host 127.0.0.1 --port 8000` (Anaconda Python + `playwright install chromium` for URLs).

---

## 9. Suggested division of talking points (team)

| Person | Own these topics |
| --- | --- |
| A | Problem, datasets, preprocess workflow, FiWi hold-out |
| B | Model architecture, fusion, loss, baselines |
| C | Metrics table, results interpretation, limitations |
| D | Demo (URL DOM vs upload), file map, how to re-run train/eval |

Everyone should still know the 60-second pitch:

> Dual-branch saliency for webpages: ResNet on the screenshot, transformer on layout/DOM regions, concat/attention fusion, U-Net heatmap. Pretrain SALICON → fine-tune UEyes webpages → test FiWi. Best model concat, CC 0.315 / AUC-Judd 0.775 on 149 pages, better than ResNet-only.

---

## 10. Limitations (say them yourselves)

1. Training structure is often **estimated**, not scraped HTML  
2. Desktop static pages only  
3. Webpage fine-tune set is relatively small → relies on SALICON transfer  
4. Not claiming SOTA against every published webpage saliency paper  
5. Demo needs Playwright Chromium installed for URL mode  

---

## 11. Quick “where is that?” index

| Need | File |
| --- | --- |
| Constants / image size | `src/config.py` |
| Download | `src/data/download.py` |
| Resize + splits + manifest | `src/data/preprocess.py` |
| OpenCV boxes | `src/data/structural.py` |
| Feature packing / hierarchy | `src/data/html_dom.py` |
| DataLoader samples | `src/data/dataset.py` |
| Neural nets + loss | `src/models/multimodal.py` |
| Metrics math | `src/metrics/saliency.py` |
| Train loop | `src/train.py` |
| FiWi scores | `src/evaluate.py` |
| Predict one image | `src/infer.py` |
| EDA plots | `src/eda.py` |
| Word results chapter | `src/report.py` |
| Website API + Playwright | `app/main.py` |
| Website UI | `app/static/*` |
| Numbers to quote | `results/RESULTS.md` |

---

*Keep this guide next to `Final_Review_Report_with_Results.docx`. Update the metrics table if you retrain.*

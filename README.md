# Webpage attention

Predicts where people look on a static desktop webpage screenshot.

ResNet reads the image, a small transformer reads layout/DOM regions, fusion draws the saliency map. SALICON pretrains the visual encoder, UEyes fine-tunes on webpages, FiWi is the held-out test.

## Setup

```powershell
$py = "$env:USERPROFILE\anaconda3\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
& $py -m playwright install chromium
```

## Run

```powershell
$py = "$env:USERPROFILE\anaconda3\python.exe"
& $py -m src.data.download
& $py -m src.data.preprocess
& $py -m src.eda
& $py -m src.train
& $py -m src.evaluate
& $py -m src.report
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — weights live in `models/best_multimodal_model.pth`.

If you only changed DOM features: `python -m src.data.rebuild_struct` then retrain.

## Docker

CPU torch image for Dokploy. Copy `models/` in before build.

```powershell
docker compose up --build
```

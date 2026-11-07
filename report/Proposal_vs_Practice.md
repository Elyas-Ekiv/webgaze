# Proposal vs Practice — What Actually Changed

Use this in evaluation when someone asks: “Is this what you proposed?”

---

## Same idea (unchanged)

- Screenshot + structure → fusion → saliency heatmap  
- SALICON pretrain → UEyes fine-tune → FiWi held-out test  
- Metrics: CC, NSS, AUC-Judd, SIM, KL-Divergence  
- Multiple baselines + multimodal variants  
- A demo so users can try the model  

---

## What’s different in practice

| Proposal said | What we actually do |
| --- | --- |
| **HTML/DOM as a main training input** | Training mostly uses **image-estimated layout regions** (OpenCV). Datasets don’t ship HTML. |
| **Image-to-HTML module** (planned / incomplete mid-report) | We have a **layout/DOM feature packer** that builds the same feature table + a small HTML dump — not a full real-page HTML parser for training. |
| **Real DOM for the model** | **Only in the URL demo**: Playwright scrapes live boxes. Upload + training ≠ live HTML. |
| **WebSaliency** for fine-tuning | **Dropped** — no reliable public download. Pipeline = SALICON + UEyes + FiWi. |
| **ResNet or ViT** as the main visual path | **Best model = ResNet + concat**. ViT is a baseline only. |
| **U-Net or upsampling decoder** | ResNet path uses **U-Net-style skip connections**; ViT / structure-only use plain upsampling. |
| **PHP + VPS** (and optional Streamlit/Flask) | **FastAPI + HTML/JS**, Docker / Dokploy-ready. |
| **Selenium + BeautifulSoup** (tools list) | **Playwright** for URLs (BeautifulSoup available). No Selenium. |
| **TensorFlow and PyTorch** | **PyTorch only**. |
| **Jupyter / Colab** as main environment | **Python scripts** (`train`, `evaluate`, etc.), not notebooks. |
| **One single “the model”** | We train **many models**, then ship the **best fusion** (`concat`) as `best_multimodal_model.pth`. |

---

## How it works in one line (practical)

**Train:** screenshot + guessed layout features → pick best fusion on FiWi.  

**Demo:**  
- Screenshot upload → guessed layout (same as training style)  
- URL paste → **real DOM boxes** via Playwright  

Same model weights, **different source for the structural features**.

---

## Key viva sentence

> The proposal describes HTML/DOM structure as the second modality. In practice we use the **same structural feature schema** everywhere. For training and screenshot uploads we fill it with **prototype layout regions from the image**. When the user pastes a URL, we fill it with **real DOM boxes** from Playwright.

---

## Why this is still valid for the project

1. Multimodal claim holds: visual + structural features, not screenshot-only.  
2. Report already notes WebSaliency was dropped and prototype layout until full HTML is available.  
3. Results show fusion (**concat**) beats ResNet-only on FiWi — structure helps.  
4. URL demo shows the intended real-DOM path end-to-end.

---

## Files to point at if asked

| Topic | File |
| --- | --- |
| Layout estimation | `src/data/structural.py` |
| Feature packing / HTML dump | `src/data/html_dom.py` |
| Live DOM from URL | `app/main.py` (`_capture`) |
| Model variants + concat | `src/models/multimodal.py` |
| Best checkpoint selection | `src/evaluate.py` |
| Numbers | `results/RESULTS.md` |

---

*Companion to `Team_Code_Guide.md` / `.docx`.*

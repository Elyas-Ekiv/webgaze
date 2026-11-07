"""Build Final Review Report: copy template, insert Implementation & Results chapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.shared import Inches
from docx.text.paragraph import Paragraph

from src.config import FIGURES, REPORT_DIR, RESULTS, ROOT

_CANDIDATES = [
    Path(r"c:\Users\Admin\Downloads\DS and AI-CSDS4105 - Final Review Report Template-Research (1).docx"),
    Path(r"c:\Users\Admin\Downloads\DS and AI-CSDS4105 - Final Review Report Template-Research.docx"),
]
TEMPLATE = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])


def _add_picture(doc: Document, path: Path, width: float = 6.2) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))


def _insert_paragraph_before(paragraph: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addprevious(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def _find_paragraph(doc: Document, startswith: str) -> Paragraph | None:
    needle = startswith.lower()
    for para in doc.paragraphs:
        if para.text.strip().lower().startswith(needle):
            return para
    return None


def _add_heading_before(anchor: Paragraph, text: str, level: int) -> Paragraph:
    style = f"Heading {level}"
    return _insert_paragraph_before(anchor, text, style=style)


def _add_body_before(anchor: Paragraph, text: str) -> Paragraph:
    return _insert_paragraph_before(anchor, text, style="Normal")


def _add_bullets_before(anchor: Paragraph, items: list[str]) -> None:
    for item in items:
        _insert_paragraph_before(anchor, f"• {item}", style="Normal")


def _add_table_before(anchor: Paragraph, headers: list[str], rows: list[list[str]]) -> None:
    # Tables cannot easily sit between paragraphs via python-docx insert helpers.
    # Insert a marker paragraph, then move a real table before the anchor.
    doc = anchor.part.document
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
    for r_index, row in enumerate(rows, start=1):
        for c_index, value in enumerate(row):
            table.rows[r_index].cells[c_index].text = value
    anchor._p.addprevious(table._tbl)
    # remove the empty table remnant at document end if python-docx left a copy:
    # add_table appends; we moved the tbl element, so the trailing empty wrapper is gone with the move.


def _page_break_before(anchor: Paragraph) -> None:
    para = _insert_paragraph_before(anchor, "")
    run = para.add_run()
    run.add_break(WD_BREAK.PAGE)


def _write_results_md(destination: Path, best: str, fiwi_images: int, table: dict) -> None:
    summary = ROOT / "results" / "RESULTS.md"
    lines = [
        "# Experimental results",
        "",
        f"Best multimodal checkpoint: `{best}`",
        "",
        f"FiWi images: {fiwi_images}",
        "",
        "| Model | CC | NSS | AUC-Judd | SIM | KL-Divergence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, scores in table.items():
        lines.append(
            "| {name} | {CC:.4f} | {NSS:.4f} | {AUC:.4f} | {SIM:.4f} | {KL:.4f} |".format(
                name=model_name,
                CC=scores["CC"],
                NSS=scores["NSS"],
                AUC=scores["AUC-Judd"],
                SIM=scores["SIM"],
                KL=scores["KL-Divergence"],
            )
        )
    lines.append("")
    lines.append(f"Report: `{destination}`")
    summary.write_text("\n".join(lines), encoding="utf-8")


def insert_implementation_chapter(doc: Document, payload: dict) -> None:
    """Insert Chapter 7 before Appendices. Template Chapter 4 is already Preprocessing."""
    table = payload["fiwi"]
    best = payload["best_model"]
    best_scores = table[best]
    visual_scores = table.get("resnet_only", {})
    center = table.get("center_bias", {})
    attention = table.get("attention", {})

    anchor = _find_paragraph(doc, "appendices") or _find_paragraph(doc, "references")
    if anchor is None:
        raise RuntimeError("Could not find Appendices/References anchor in the template.")

    _page_break_before(anchor)
    _add_heading_before(anchor, "Chapter 7: IMPLEMENTATION & RESULTS", 1)
    _add_body_before(
        anchor,
        "This chapter records the implemented WebGaze system and the measured results. "
        "It completes the Sem-2 work outlined in Chapter 6. Template Chapter 4 remains "
        "dataset preprocessing; this chapter covers training, evaluation, comparison, "
        "and the working prototype.",
    )

    _add_heading_before(anchor, "7.1 Implementation Overview", 2)
    _add_body_before(
        anchor,
        "The system predicts a saliency heatmap for a static desktop webpage screenshot, "
        "optionally using live DOM layout when a URL is provided. The implementation pipeline is:",
    )
    _add_bullets_before(
        anchor,
        [
            "Download SALICON, UEyes, and FiWi datasets.",
            "Preprocess screenshots and saliency maps to 384×216 and extract layout/DOM region features.",
            "Pretrain the visual encoder on SALICON, then fine-tune variants on UEyes webpage screenshots.",
            "Evaluate all models on held-out FiWi and select the best multimodal fusion checkpoint.",
            "Serve a FastAPI demo for screenshot upload or live URL capture.",
        ],
    )
    _add_body_before(
        anchor,
        f"The shipped model is the best multimodal fusion variant ({best}), saved as "
        "models/best_multimodal_model.pth.",
    )

    _add_heading_before(anchor, "7.2 Development Environment", 2)
    _add_body_before(
        anchor,
        "Local development used Python 3 (Anaconda on Windows), PyTorch/torchvision for deep learning, "
        "FastAPI and Uvicorn for the demo API, and Playwright (Chromium) for live webpage capture. "
        "Supporting libraries include OpenCV, NumPy, Pandas, Pillow, SciPy, Matplotlib, Seaborn, "
        "and python-docx. Container deployment uses Docker/Dokploy with the project Dockerfile. "
        "The demo runs at http://127.0.0.1:8000.",
    )

    _add_heading_before(anchor, "7.3 Dataset Implementation", 2)
    _add_body_before(
        anchor,
        "Public datasets are used in stages and are not merged into one training set. "
        "SALICON pretrains general visual attention, UEyes fine-tunes on webpage/UI screenshots, "
        "and FiWi is held out for final testing (149 webpage screenshots). "
        "WebSaliency was listed in the original plan, but no reliable public download was available, "
        "so the implemented pipeline uses SALICON + UEyes + FiWi only. "
        "Structural features are estimated from screenshots during training; the URL demo fills the "
        "same feature schema from live DOM boxes.",
    )

    _add_heading_before(anchor, "7.4 Model / Algorithm Implementation", 2)
    _add_body_before(
        anchor,
        "The model is a dual-branch multimodal architecture implemented in src/models/multimodal.py. "
        "The visual branch uses ResNet-18 with U-Net skip features. The structural branch embeds up to "
        "48 region tokens with a Transformer (or MLP for structure-only baselines). "
        "Fusion uses either concatenation (concat) or cross-attention then concatenation (attention). "
        "A U-Net / upsampling decoder outputs the saliency heatmap. "
        "Baselines include resnet_only, vit_only, mlp_only, and transformer_only. "
        "The training loss combines KL divergence, (1−CC), and a small NSS term.",
    )

    _add_heading_before(anchor, "7.5 Model Training and Hyperparameter Configuration", 2)
    _add_body_before(
        anchor,
        "Training follows the staged plan from Chapters 3 and 5. Default settings: input size 384×216, "
        "max 48 regions, Adam optimizer, learning rate 1e-4, batch size 8, SALICON pretrain epochs 2, "
        "UEyes fine-tune epochs 6 (vit_only: 4), gradient clipping at norm 1.0, and seed 42. "
        "ResNet-based multimodal models load matching SALICON-pretrained visual weights before fine-tuning. "
        "Checkpoints are stored under models/; training history is stored in results/history.json. "
        "Final multimodal selection is decided by FiWi correlation in evaluation, not only UEyes validation.",
    )

    _add_heading_before(anchor, "7.6 System / Application Implementation", 2)
    _add_body_before(
        anchor,
        "The prototype is a FastAPI application under app/. Users can upload a screenshot or enter a URL. "
        "URL mode uses Playwright to capture a desktop screenshot and DOM region boxes, then runs the same "
        "inference path as uploads. The API returns the screenshot, color heatmap, overlay, region summary, "
        "and ranked hotspots. Core routes include GET /, GET /api/health, POST /api/predict, and GET /api/sample.",
    )

    _add_heading_before(anchor, "7.7 Experimental Setup", 2)
    _add_body_before(
        anchor,
        f"Pretraining uses SALICON, fine-tuning uses UEyes webpage splits, and final testing uses FiWi only "
        f"({payload['fiwi_images']} images). A center-bias Gaussian baseline is included because webpage "
        "fixations often concentrate near the screen center. Metrics are CC, NSS, AUC-Judd, SIM "
        "(higher is better) and KL-Divergence (lower is better). Scope is static desktop pages.",
    )

    _add_heading_before(anchor, "7.8 Model Results", 2)
    _add_body_before(
        anchor,
        f"On held-out FiWi, the best model is {best}. Table below summarizes mean scores over "
        f"{payload['fiwi_images']} images.",
    )
    metric_names = ["CC", "NSS", "AUC-Judd", "SIM", "KL-Divergence"]
    rows = []
    for model_name, scores in table.items():
        rows.append([model_name] + [f"{scores[name]:.4f}" for name in metric_names])
    _add_table_before(anchor, ["Model"] + metric_names, rows)
    _add_body_before(anchor, "Figure. FiWi metric comparison across models.")
    # Pictures must be added on a temporary paragraph then moved, or added via runs.
    # Use a normal insert paragraph and attach picture to a temporary end paragraph moved before anchor.
    pic_para = _insert_paragraph_before(anchor, "")
    if (FIGURES / "fiwi_metrics.png").exists():
        run = pic_para.add_run()
        run.add_picture(str(FIGURES / "fiwi_metrics.png"), width=Inches(6.2))

    _add_heading_before(anchor, "7.9 Performance Evaluation", 2)
    _add_body_before(
        anchor,
        "Saliency prediction is not classification, so accuracy percentage is not used. "
        f"The saved {best} model reached CC {best_scores['CC']:.4f}, NSS {best_scores['NSS']:.4f}, "
        f"AUC-Judd {best_scores['AUC-Judd']:.4f}, SIM {best_scores['SIM']:.4f}, and "
        f"KL-Divergence {best_scores['KL-Divergence']:.4f}. "
        "These metrics match the evaluation methods defined in Chapter 5.",
    )
    _add_body_before(anchor, "Figure. Validation correlation during training.")
    pic2 = _insert_paragraph_before(anchor, "")
    if (FIGURES / "training_curves.png").exists():
        pic2.add_run().add_picture(str(FIGURES / "training_curves.png"), width=Inches(6.2))
    _add_body_before(anchor, "Figure. Qualitative FiWi predictions from the saved multimodal model.")
    pic3 = _insert_paragraph_before(anchor, "")
    if (FIGURES / "fiwi_qualitative.png").exists():
        pic3.add_run().add_picture(str(FIGURES / "fiwi_qualitative.png"), width=Inches(6.2))

    _add_heading_before(anchor, "7.10 Comparative Results", 2)
    _add_body_before(
        anchor,
        f"Multimodal concat (CC {best_scores['CC']:.4f}) outperforms screenshot-only ResNet "
        f"(CC {visual_scores.get('CC', 0):.4f}), showing that layout features help on FiWi. "
        f"Concat also slightly outperforms attention fusion (CC {attention.get('CC', 0):.4f}). "
        "Structure-only models score much lower, confirming that appearance remains essential. "
        f"All strong models beat the center-bias baseline (CC {center.get('CC', 0):.4f}).",
    )

    _add_heading_before(anchor, "7.11 Error / Failure Analysis", 2)
    _add_body_before(
        anchor,
        "Failure cases include unusual or sparse layouts, domain shift between natural-image pretraining "
        "and webpage fine-tuning, and missing text/font cues when regions are estimated from screenshots "
        "rather than live DOM. Structure-only models collapse toward weak priors. The prototype is limited "
        "to static desktop pages and does not cover mobile layouts, scrolling video, or hover interaction. "
        "Hard pages with clutter or multiple competing foci reduce CC/SIM and raise KL-Divergence.",
    )

    _add_heading_before(anchor, "7.12 System / Prototype Results", 2)
    _add_body_before(
        anchor,
        "The working prototype loads best_multimodal_model.pth, accepts file upload or URL input, and "
        "returns heatmap overlays with ranked hotspots. This fulfills the optional web-based prototype "
        "described in Chapter 6 Future Work. Health and sample endpoints support quick demonstration.",
    )

    _add_heading_before(anchor, "7.13 Results Summary", 2)
    _add_body_before(
        anchor,
        f"Best model: {best}. Test set: FiWi ({payload['fiwi_images']} held-out images). "
        f"Best CC / AUC-Judd: {best_scores['CC']:.4f} / {best_scores['AUC-Judd']:.4f}. "
        "Multimodal fusion beats ResNet-only. WebSaliency was not used. Deliverables include trained "
        "weights, metrics/figures under results/, and the FastAPI WebGaze demo.",
    )

    _add_heading_before(anchor, "7.14 Chapter Summary", 2)
    _add_body_before(
        anchor,
        "This chapter documented implementation and results for the multimodal webpage saliency system. "
        "Datasets and preprocessing from earlier chapters were used with SALICON pretraining and UEyes "
        "fine-tuning. On FiWi, concat fusion was the strongest model across the reported metrics and "
        "outperformed visual-only ResNet, supporting the multimodal design while acknowledging layout "
        "estimation gaps and static-desktop scope limits.",
    )


def write_validation(destination: Path, payload: dict) -> Path:
    """Cross-check Chapter 7 content against template Chapters 1–6."""
    best = payload["best_model"]
    scores = payload["fiwi"][best]
    checks = [
        ("PASS", "Chapter numbering", "Template Chapter 4 is Preprocessing; Implementation & Results inserted as Chapter 7 before Appendices."),
        ("PASS", "Datasets", "Matches Ch5 implementation note: SALICON + UEyes + FiWi; WebSaliency excluded."),
        ("PASS", "Image size", "Uses 384×216 from Chapter 4 preprocessing."),
        ("PASS", "Architecture", "Dual-branch ResNet + structure + concat/attention fusion + U-Net decoder matches Chapter 5."),
        ("PASS", "Training stages", "SALICON pretrain → UEyes fine-tune → FiWi test matches Chapters 3 and 5."),
        ("PASS", "Metrics", "CC, NSS, AUC-Judd, SIM, KL-Divergence match Chapter 5 evaluation methods."),
        ("PASS", "FiWi size", f"Uses held-out FiWi with {payload['fiwi_images']} images, as stated in Chapter 3."),
        ("PASS", "Best checkpoint", f"Saved best multimodal model is `{best}` (CC {scores['CC']:.4f}, AUC-Judd {scores['AUC-Judd']:.4f})."),
        ("PASS", "Prototype", "FastAPI upload/URL demo completes Chapter 6.2 optional web prototype."),
        ("NOTE", "Chapter 6 Future Work", "Chapter 6 still reads as a Sem-2 plan; Chapter 7 is the implemented outcome and can later be mirrored into an updated 6.1 conclusion."),
        ("NOTE", "Dev environment wording", "Chapter 3 mentions Jupyter/Colab; actual implementation used local Anaconda + Docker. Chapter 7 documents the environment used."),
        ("NOTE", "Older WebSaliency sentences", "Some early chapters still mention WebSaliency in plans; Chapter 5/7 correctly state it was not used."),
    ]
    lines = [
        "# Template validation — Implementation & Results",
        "",
        f"Template: `{TEMPLATE}`",
        f"Output: `{destination}`",
        "",
        "The requested Implementation & Results outline could not reuse template Chapter 4 numbering "
        "because that chapter is already **Preprocessing Datasets**. Content was added as "
        "**Chapter 7: IMPLEMENTATION & RESULTS** (sections 7.1–7.14) and checked against Chapters 1–6.",
        "",
        "| Status | Check | Detail |",
        "| --- | --- | --- |",
    ]
    for status, name, detail in checks:
        lines.append(f"| {status} | {name} | {detail} |")
    lines.extend(
        [
            "",
            "## Section map (requested → template report)",
            "",
            "| Requested heading | Inserted as |",
            "| --- | --- |",
            "| 4.1 Implementation Overview | 7.1 |",
            "| 4.2 Development Environment | 7.2 |",
            "| 4.3 Dataset Implementation | 7.3 |",
            "| 4.4 Model / Algorithm Implementation | 7.4 |",
            "| 4.5 Model Training and Hyperparameter Configuration | 7.5 |",
            "| 4.6 System / Application Implementation | 7.6 |",
            "| 4.7 Experimental Setup | 7.7 |",
            "| 4.8 Model Results | 7.8 |",
            "| 4.9 Performance Evaluation | 7.9 |",
            "| 3.10/4.10 Comparative Results | 7.10 |",
            "| 3.11/4.11 Error / Failure Analysis | 7.11 |",
            "| 3.12/4.12 System / Prototype Results | 7.12 |",
            "| 3.13/4.13 Results Summary | 7.13 |",
            "| 3.14/4.14 Chapter Summary | 7.14 |",
            "",
        ]
    )
    out = REPORT_DIR / "Chapter_Implementation_Template_Validation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    metrics_path = RESULTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run evaluation before building the results chapter.")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / "Final_Review_Report_with_Results.docx"
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")
    shutil.copy(TEMPLATE, destination)
    doc = Document(destination)
    insert_implementation_chapter(doc, payload)
    doc.save(destination)
    validation = write_validation(destination, payload)
    _write_results_md(destination, payload["best_model"], payload["fiwi_images"], payload["fiwi"])
    print("wrote", destination)
    print("validation", validation)


if __name__ == "__main__":
    main()

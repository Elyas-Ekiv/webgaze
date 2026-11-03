"""Append a results chapter to a copy of the course review."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Inches

from src.config import FIGURES, REPORT_DIR, RESULTS, ROOT

_CANDIDATES = [
    Path(r"c:\Users\Admin\Downloads\DS and AI-CSDS4105 - Final Review Report Template-Research (1).docx"),
    Path(r"c:\Users\Admin\Downloads\DS and AI-CSDS4105 - Final Review Report Template-Research.docx"),
]
TEMPLATE = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])


def _add_picture(doc: Document, path: Path, width: float = 6.2) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))


def main() -> None:
    metrics_path = RESULTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run evaluation before building the results chapter.")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = payload["fiwi"]
    best = payload["best_model"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / "Final_Review_Report_with_Results.docx"
    if TEMPLATE.exists():
        shutil.copy(TEMPLATE, destination)
        doc = Document(destination)
    else:
        doc = Document()

    doc.add_heading("Chapter 7: Experimental Results", level=1)
    doc.add_paragraph(
        "This chapter records the implemented system. SALICON was used to pretrain the visual encoder, "
        "UEyes webpage screenshots were used to fine-tune the models, and FiWi was held out for the final test. "
        "WebSaliency was not used because no reliable public download was available. "
        "Screenshots and saliency maps were resized to 384 by 216. Layout features were extracted from each "
        "screenshot for the structural branch. When the demo is given a URL, the same feature columns are filled "
        f"from the live DOM. The saved model is the {best} fusion variant, chosen by FiWi correlation."
    )
    doc.add_paragraph(
        f"The held-out FiWi test contains {payload['fiwi_images']} webpage screenshots. "
        "CC, NSS, AUC-Judd, and SIM are higher when the prediction is closer to human attention. "
        "KL-Divergence is lower when the predicted distribution is closer to the human map."
    )

    metric_names = ["CC", "NSS", "AUC-Judd", "SIM", "KL-Divergence"]
    grid = doc.add_table(rows=1, cols=1 + len(metric_names))
    header = grid.rows[0].cells
    header[0].text = "Model"
    for index, name in enumerate(metric_names, start=1):
        header[index].text = name
    for model_name, scores in table.items():
        cells = grid.add_row().cells
        cells[0].text = model_name
        for index, name in enumerate(metric_names, start=1):
            cells[index].text = f"{scores[name]:.4f}"

    doc.add_paragraph("Figure. FiWi metric comparison.")
    _add_picture(doc, FIGURES / "fiwi_metrics.png")
    doc.add_paragraph("Figure. Validation correlation while training.")
    _add_picture(doc, FIGURES / "training_curves.png")
    doc.add_paragraph("Figure. Qualitative FiWi predictions from the saved multimodal model.")
    _add_picture(doc, FIGURES / "fiwi_qualitative.png")
    doc.add_paragraph("Figure. Dataset counts and image-size distributions from preprocessing.")
    _add_picture(doc, FIGURES / "dataset_file_counts.png", 5.2)
    _add_picture(doc, FIGURES / "image_width_distribution.png", 5.2)
    _add_picture(doc, FIGURES / "image_height_distribution.png", 5.2)
    best_scores = table[best]
    visual_scores = table.get("resnet_only", {})
    doc.add_paragraph(
        f"On FiWi, the saved {best} model reached CC {best_scores['CC']:.4f}, "
        f"NSS {best_scores['NSS']:.4f}, AUC-Judd {best_scores['AUC-Judd']:.4f}, "
        f"SIM {best_scores['SIM']:.4f}, and KL-Divergence {best_scores['KL-Divergence']:.4f}. "
        f"The screenshot-only ResNet reached CC {visual_scores.get('CC', 0):.4f}. "
        "The center-bias baseline is included because webpage fixations often fall near the middle of the screen. "
        "Layout-only models score lower, which matches the idea that structure helps most when it is fused with the screenshot."
    )
    doc.save(destination)
    summary = ROOT / "results" / "RESULTS.md"
    lines = [
        "# Experimental results",
        "",
        f"Best multimodal checkpoint: `{best}`",
        "",
        f"FiWi images: {payload['fiwi_images']}",
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
    print("wrote", destination)


if __name__ == "__main__":
    main()

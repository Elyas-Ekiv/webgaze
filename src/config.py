from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

IMG_H, IMG_W = 216, 384
MAX_REGIONS = 48

NUMERIC_FEATURES = [
    "x",
    "y",
    "width",
    "height",
    "area",
    "relative_area",
    "center_x",
    "center_y",
    "aspect_ratio",
    "brightness",
    "contrast",
    "edge_density",
    "non_white_ratio",
    "text_length",
    "font_size",
    "depth",
    "n_children",
    "parent_area_ratio",
]
ELEMENT_TYPES = ["header", "background", "text_or_content", "layout_region"]
STRUCT_DIM = len(NUMERIC_FEATURES) + len(ELEMENT_TYPES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

UEYES_MD5 = "c2d53e6af0a47e1f459416d6839ec2c1"
UEYES_BYTES = 12_912_746_325
FIWI_DARUS_MD5 = "afa0e238a37688d9e51620a02e77df78"
FIWI_DARUS_BYTES = 217_776_719

SALICON_DRIVE_IDS = {
    "images": "1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5",
    "fixations": "1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj",
    "saliency": "1PnO7szbdub1559LfjYHMy65EDC4VhJC8",
}

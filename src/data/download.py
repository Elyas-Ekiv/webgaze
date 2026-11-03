"""Download SALICON, UEyes, and FiWi into data/raw."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.config import (
    DATA_RAW,
    FIWI_DARUS_BYTES,
    FIWI_DARUS_MD5,
    SALICON_DRIVE_IDS,
    UEYES_BYTES,
    UEYES_MD5,
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curl(url: str, dest: Path, expected_bytes: int | None = None) -> None:
    if expected_bytes and expected_bytes >= 50_000_000:
        _parallel_download(url, dest, expected_bytes)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_bytes and dest.stat().st_size == expected_bytes:
        print(f"already downloaded {dest.name}")
        return
    cmd = [
        "curl.exe",
        "-L",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    print(" ".join(cmd[:6]), dest.name)
    subprocess.check_call(cmd)


def _parallel_download(url: str, dest: Path, expected_bytes: int, connections: int = 8) -> None:
    """Download a large file in ranged chunks. A finished prefix from curl is kept."""
    import json
    import threading
    import time

    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    parts_path = dest.with_suffix(dest.suffix + ".parts")
    if dest.exists() and dest.stat().st_size == expected_bytes and not parts_path.exists():
        print(f"already downloaded {dest.name}")
        return

    done: list[list[int]] = []
    if parts_path.exists():
        done = json.loads(parts_path.read_text(encoding="utf-8"))
    elif dest.exists() and dest.stat().st_size < expected_bytes:
        done = [[0, dest.stat().st_size]]

    with dest.open("r+b" if dest.exists() else "wb") as handle:
        handle.truncate(expected_bytes)

    chunk = 32 * 1024 * 1024
    pending: list[tuple[int, int]] = []
    pos = 0
    while pos < expected_bytes:
        end = min(pos + chunk, expected_bytes)
        covered = any(start <= pos and end <= stop for start, stop in done)
        if not covered:
            pending.append((pos, end - 1))
        pos = end

    if not pending:
        print(f"already downloaded {dest.name}")
        parts_path.unlink(missing_ok=True)
        return

    print(f"downloading {dest.name}: {len(pending)} chunks, {connections} connections")
    lock = threading.Lock()
    finished = 0

    def fetch(span: tuple[int, int]) -> tuple[int, int]:
        start, end = span
        last_error = None
        for _ in range(6):
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 180),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"expected 206, got {response.status_code}")
                    data = response.content
                    if len(data) != end - start + 1:
                        raise RuntimeError(f"short read {len(data)}")
                with lock:
                    with dest.open("r+b") as handle:
                        handle.seek(start)
                        handle.write(data)
                return start, end + 1
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise RuntimeError(f"chunk {start}-{end} failed: {last_error}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(fetch, span) for span in pending]
        for future in as_completed(futures):
            start, stop = future.result()
            with lock:
                done.append([start, stop])
                parts_path.write_text(json.dumps(done), encoding="utf-8")
                finished += 1
                if finished % 5 == 0 or finished == len(pending):
                    got = min(expected_bytes, sum(stop - start for start, stop in done))
                    print(f"{dest.name} {got / 1e9:.2f} / {expected_bytes / 1e9:.2f} GB")
    parts_path.unlink(missing_ok=True)


def _stamp_ok(path: Path, expected: str) -> bool:
    stamp = path.with_suffix(path.suffix + ".md5ok")
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == expected and path.exists():
        return True
    if not path.exists():
        return False
    print(f"checking md5 for {path.name}")
    actual = _md5(path)
    if actual != expected:
        raise RuntimeError(f"{path.name} md5 {actual} does not match {expected}")
    stamp.write_text(expected, encoding="utf-8")
    return True


def _extract_zip(zip_path: Path, dest: Path) -> None:
    marker = dest / ".extracted"
    if marker.exists():
        print(f"already extracted {dest.name}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    marker.write_text("ok", encoding="utf-8")


def download_ueyes() -> None:
    dest = DATA_RAW / "UEyes_dataset.zip"
    _curl(
        "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1",
        dest,
        UEYES_BYTES,
    )
    _stamp_ok(dest, UEYES_MD5)
    _extract_zip(dest, DATA_RAW / "ueyes")


def download_fiwi() -> None:
    dest = DATA_RAW / "fiwi_webpage_dataset.zip"
    out = DATA_RAW / "fiwi"
    if (out / ".extracted").exists():
        print("FiWi already extracted")
        return
    try:
        _curl(
            "https://www.dropbox.com/s/30nxg2uwd1wpb80/webpage_dataset.zip?dl=1",
            dest,
            279_476_626,
        )
        _extract_zip(dest, out)
        return
    except Exception as exc:
        print(f"official FiWi link failed ({exc}); using DARUS preprocessed archive")
    darus = DATA_RAW / "FiWI_darus.zip"
    _curl(
        "https://darus.uni-stuttgart.de/api/access/datafile/171576",
        darus,
        FIWI_DARUS_BYTES,
    )
    _stamp_ok(darus, FIWI_DARUS_MD5)
    _extract_zip(darus, out)


def _gdown(file_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already downloaded {dest.name}")
        return
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"SALICON download failed for {dest.name}")


def _move_images(src_root: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    for path in src_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        if "test" in {part.lower() for part in path.parts}:
            continue
        split = "val" if "val" in {part.lower() for part in path.parts} else "train"
        target = dest_root / split / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.move(str(path), str(target))


def download_salicon() -> None:
    root = DATA_RAW / "salicon"
    if (root / ".extracted").exists():
        print("SALICON already extracted")
        return
    tmp = DATA_RAW / "salicon_zips"
    tmp.mkdir(parents=True, exist_ok=True)
    images_zip = tmp / "salicon_images.zip"
    fix_zip = tmp / "salicon_fixations.zip"
    sal_zip = tmp / "salicon_saliency.zip"
    _gdown(SALICON_DRIVE_IDS["images"], images_zip)
    _gdown(SALICON_DRIVE_IDS["fixations"], fix_zip)
    _gdown(SALICON_DRIVE_IDS["saliency"], sal_zip)

    stage = tmp / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    for archive, folder in ((images_zip, "images"), (fix_zip, "fixations"), (sal_zip, "saliency")):
        print(f"extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(stage / folder)
    _move_images(stage / "images", root / "stimuli")
    _move_images(stage / "fixations", root / "fixations")
    _move_images(stage / "saliency", root / "saliency")
    (root / ".extracted").write_text("ok", encoding="utf-8")
    print(
        "SALICON stimuli",
        len(list((root / "stimuli").rglob("*"))),
        "saliency",
        len(list((root / "saliency").rglob("*"))),
    )


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    download_fiwi()
    download_salicon()
    download_ueyes()


if __name__ == "__main__":
    main()

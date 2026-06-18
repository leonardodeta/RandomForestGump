"""
crop_faces.py
-------------
Reusable MTCNN face-cropping utilities plus a command-line interface.

For every image in the input folder:
  - if a face is detected, the highest-confidence face is cropped;
  - if no face is detected, the original image is resized as a fallback.

The relative folder structure is preserved, so the function works for both flat
folders and ImageFolder-style datasets.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from facenet_pytorch import MTCNN
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CropStats:
    total: int = 0
    found: int = 0
    not_found: int = 0
    failed: int = 0


def get_image_paths(folder: Path) -> List[Path]:
    """Collect all supported images in a flat or nested folder."""
    folder = Path(folder)
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def _resize_fallback(img: Image.Image, image_size: int) -> Image.Image:
    return img.resize((image_size, image_size), Image.BILINEAR)


def crop_image(img: Image.Image, mtcnn: MTCNN, image_size: int) -> tuple[Image.Image, bool]:
    """Return ``(cropped_image, face_found)`` for one PIL image."""
    img = img.convert("RGB")
    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0 or probs is None:
        return _resize_fallback(img, image_size), False

    best_idx = int(probs.argmax())
    box = boxes[best_idx]

    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = img.size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return _resize_fallback(img, image_size), False

    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((image_size, image_size), Image.BILINEAR)
    return crop, True


def crop_and_save(
    img_path: Path,
    out_path: Path,
    mtcnn: MTCNN,
    image_size: int,
) -> bool:
    """Crop one image and save it. Returns True if a face was found."""
    with Image.open(img_path) as img:
        crop, face_found = crop_image(img, mtcnn, image_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return face_found


def build_mtcnn(image_size: int, device: Optional[torch.device] = None) -> MTCNN:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MTCNN(
        image_size=image_size,
        margin=0,
        keep_all=False,
        device=device,
        post_process=False,
    )


def crop_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    image_size: int = 160,
    *,
    device: Optional[torch.device] = None,
    log_every: int = 100,
) -> CropStats:
    """Crop all images from ``input_folder`` into ``output_folder``."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = build_mtcnn(image_size=image_size, device=device)

    img_paths = get_image_paths(input_folder)
    stats = CropStats(total=len(img_paths))

    print(f"[crop_faces] device     = {device}")
    print(f"[crop_faces] input      = {input_folder}")
    print(f"[crop_faces] output     = {output_folder}")
    print(f"[crop_faces] image-size = {image_size}")
    print(f"[crop_faces] images     = {len(img_paths)}")

    for i, img_path in enumerate(img_paths, start=1):
        rel_path = img_path.relative_to(input_folder)
        out_path = output_folder / rel_path

        try:
            face_found = crop_and_save(img_path, out_path, mtcnn, image_size)
            if face_found:
                stats.found += 1
            else:
                stats.not_found += 1
        except Exception as exc:
            print(f"  [WARN] {img_path}: {exc}")
            stats.failed += 1

        if log_every > 0 and i % log_every == 0:
            print(
                f"  [{i}/{len(img_paths)}] faces: {stats.found} | "
                f"fallbacks: {stats.not_found} | failed: {stats.failed}"
            )

    print("\n[crop_faces] Completed.")
    print(f"  Faces found: {stats.found} / {stats.total}")
    print(f"  Fallbacks:   {stats.not_found} / {stats.total}")
    print(f"  Failed:      {stats.failed} / {stats.total}")
    print(f"  Output in:   {output_folder}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Crop faces in a folder using MTCNN."
    )
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output folder for crops")
    parser.add_argument("--image-size", type=int, default=160, help="Output side length")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N images")
    # Kept only for backward compatibility with old commands. MTCNN is run in a
    # single process because GPU-backed multiprocessing is fragile on many exam VMs.
    parser.add_argument("--workers", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    crop_folder(
        input_folder=args.input,
        output_folder=args.output,
        image_size=args.image_size,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

"""
run_competition.py
------------------
Run the final retrieval pipeline and optionally submit results.

Default behaviour is intentionally defensible: pretrained/fine-tuned FaceNet
embeddings, L2 normalization, cosine similarity, and horizontal-flip TTA. Heavy
post-processing options are explicit flags and should be enabled only if
validation ablations support them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from PIL import Image

from crop_faces import crop_folder
from facenet_encoder import FaceNetEncoder
from search_system import RetrievalSystem


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_folder(folder: str | Path) -> Tuple[List[Image.Image], List[str]]:
    """Load all images from a flat folder while preserving sorted filenames."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    images, filenames = [], []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(VALID_EXTENSIONS):
            path = folder / filename
            if path.is_file():
                with Image.open(path) as img:
                    images.append(img.convert("RGB").copy())
                filenames.append(filename)
    return images, filenames


def submit(results: dict, groupname: str, url: str, timeout: int = 30) -> None:
    payload = {"groupname": groupname, "images": results}
    response = requests.post(url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[submit] HTTP error: {exc}")
        print(f"[submit] Server response: {response.text}")
        raise

    try:
        out = response.json()
    except json.JSONDecodeError:
        print(f"[submit] Non-JSON response: {response.text}")
        return

    print(f"[submit] response = {out}")
    if "accuracy" in out:
        print(f"[submit] accuracy = {out['accuracy']}")


def _prepare_auto_cropped_data(
    data_folder: Path,
    output_folder: Path,
    image_size: int,
    device: torch.device,
) -> Path:
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    crop_folder(data_folder / "query", output_folder / "query", image_size=image_size, device=device, log_every=100)
    crop_folder(data_folder / "gallery", output_folder / "gallery", image_size=image_size, device=device, log_every=100)
    return output_folder


def _expand_mmr_schedule(values: List[float] | None, top_k: int = 10) -> List[float] | None:
    if not values:
        return None
    for value in values:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("All --mmr-lambda values must be in [0, 1]")
    if len(values) == 1:
        return values * top_k
    return (values * (top_k // len(values) + 1))[:top_k]


def validate_submission(results: dict[str, list[str]], query_filenames: list[str], gallery_filenames: list[str], top_k: int) -> None:
    expected_queries = set(query_filenames)
    actual_queries = set(results)
    if actual_queries != expected_queries:
        missing = sorted(expected_queries - actual_queries)
        extra = sorted(actual_queries - expected_queries)
        raise ValueError(f"Submission query keys mismatch. Missing={missing[:5]}, extra={extra[:5]}")

    gallery_set = set(gallery_filenames)
    for qfn, gfns in results.items():
        if len(gfns) != top_k:
            raise ValueError(f"Query {qfn} has {len(gfns)} results; expected {top_k}")
        if len(set(gfns)) != len(gfns):
            raise ValueError(f"Query {qfn} has duplicate gallery results: {gfns}")
        unknown = [name for name in gfns if name not in gallery_set]
        if unknown:
            raise ValueError(f"Query {qfn} contains unknown gallery filenames: {unknown[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True, help="Folder containing query/ and gallery/")
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--submit-url", default="http://localhost:3001/retrieval/")

    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal flip TTA")
    parser.add_argument("--kreciprocal", action="store_true", help="Enable k-reciprocal re-ranking")
    parser.add_argument("--no-kreciprocal", action="store_true", help="Legacy compatibility flag; keeps k-reciprocal disabled")
    parser.add_argument("--qe", action="store_true", help="Enable alpha query expansion")
    parser.add_argument("--mmr", action="store_true", help="Enable MMR diversification; use only if validation improves")
    parser.add_argument("--dry-run", action="store_true", help="Do not submit; save submission JSON locally")

    parser.add_argument("--checkpoint", default=None, help="Optional fine-tuned checkpoint")
    parser.add_argument("--arch", default="inception_resnet_v1", choices=["inception_resnet_v1", "inception_resnet_v2"])

    parser.add_argument("--qe-top-k", type=int, default=5)
    parser.add_argument("--qe-alpha", type=float, default=3.0)
    parser.add_argument("--mmr-lambda", type=float, nargs="+", default=None, help="MMR lambda schedule. One value is repeated for all ranks")
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--kr-lambda", type=float, default=0.3)
    parser.add_argument("--max-kreciprocal-matrix-elements", type=int, default=50_000_000)
    parser.add_argument("--mmr-pool", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10, help="Number of gallery images per query")

    parser.add_argument("--auto-crop", action="store_true", help="Crop query/gallery with MTCNN before retrieval")
    parser.add_argument("--crop-image-size", type=int, default=None, help="Crop size. Defaults to encoder input size")
    parser.add_argument("--crop-output-folder", default=".cropped_competition")
    parser.add_argument("--output-json", default="submission.json")
    parser.add_argument("--output-config", default="submission_config.json")

    args = parser.parse_args()

    if args.kreciprocal and args.no_kreciprocal:
        raise ValueError("Use either --kreciprocal or --no-kreciprocal, not both")

    device = pick_device()
    print(f"[main] device = {device}")

    encoder = FaceNetEncoder(device=device, checkpoint_path=args.checkpoint, arch=args.arch)

    data_folder = Path(args.data_folder)
    if args.auto_crop:
        crop_size = args.crop_image_size or encoder.image_size
        data_folder = _prepare_auto_cropped_data(
            data_folder=data_folder,
            output_folder=Path(args.crop_output_folder),
            image_size=crop_size,
            device=device,
        )

    query_images, query_filenames = load_folder(data_folder / "query")
    gallery_images, gallery_filenames = load_folder(data_folder / "gallery")
    print(f"[main] query={len(query_images)} gallery={len(gallery_images)}")

    if not query_images:
        raise ValueError(f"No query images found in {data_folder / 'query'}")
    if len(gallery_images) < args.top_k:
        raise ValueError(f"The gallery must contain at least {args.top_k} images for submission")

    system = RetrievalSystem(
        encoder=encoder,
        use_tta=not args.no_tta,
        use_kreciprocal=args.kreciprocal,
        k1=args.k1,
        k2=args.k2,
        kr_lambda=args.kr_lambda,
        max_kreciprocal_matrix_elements=args.max_kreciprocal_matrix_elements,
        use_qe=args.qe,
        qe_top_k=args.qe_top_k,
        qe_alpha=args.qe_alpha,
        use_mmr=args.mmr,
        mmr_initial_pool=args.mmr_pool,
        mmr_lambda_schedule=_expand_mmr_schedule(args.mmr_lambda, top_k=args.top_k),
        top_k_output=args.top_k,
    )

    results = system.run(query_images, query_filenames, gallery_images, gallery_filenames)
    validate_submission(results, query_filenames, gallery_filenames, top_k=args.top_k)

    config_payload = {
        "groupname": args.group_name,
        "encoder_arch": encoder.arch,
        "encoder_image_size": encoder.image_size,
        "checkpoint": args.checkpoint,
        "auto_crop": args.auto_crop,
        "retrieval_config": system.config.as_dict(),
    }

    if args.dry_run:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"groupname": args.group_name, "images": results}, f, indent=2)
        with open(args.output_config, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
        print(f"[main] dry-run, saved to {args.output_json}")
        print(f"[main] config saved to {args.output_config}")
    else:
        submit(results, args.group_name, args.submit_url)


if __name__ == "__main__":
    main()

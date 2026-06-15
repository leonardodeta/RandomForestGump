"""
run_competition.py
------------------
Script to run the final retrieval pipeline and optionally submit results.

Typical usage:
    python run_competition.py \
        --data-folder /path/to/test_data \
        --group-name "random_forest_gump" \
        --submit-url http://localhost:3001/retrieval/

Dry run:
    python run_competition.py \
        --data-folder /path/to/test_data \
        --group-name "random_forest_gump" \
        --dry-run
"""

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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_folder(folder: str | Path) -> Tuple[List[Image.Image], List[str]]:
    """Load all images from a flat folder while preserving filenames."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    images, filenames = [], []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(VALID_EXTENSIONS):
            path = folder / filename
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
            filenames.append(filename)
    return images, filenames


def submit(results: dict, groupname: str, url: str, timeout: int = 30) -> None:
    """Submit JSON to the competition server with explicit error handling."""
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
    """Create cropped query/gallery folders and return their parent folder."""
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    crop_folder(
        data_folder / "query",
        output_folder / "query",
        image_size=image_size,
        device=device,
        log_every=100,
    )
    crop_folder(
        data_folder / "gallery",
        output_folder / "gallery",
        image_size=image_size,
        device=device,
        log_every=100,
    )
    return output_folder


def _expand_mmr_schedule(values: List[float] | None) -> List[float] | None:
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values * 10
    return (values * (10 // len(values) + 1))[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True,
                        help="Folder containing query/ and gallery/")
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--submit-url",
                        default="http://localhost:3001/retrieval/")

    # Component toggles. Defaults are conservative and validation-friendly.
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable horizontal flip TTA")
    parser.add_argument("--no-kreciprocal", action="store_true",
                        help="Disable k-reciprocal re-ranking")
    parser.add_argument("--qe", action="store_true", default=False,
                        help="Enable alpha query expansion")
    parser.add_argument("--qe-top-k", type=int, default=5,
                        help="Number of neighbours for alpha-QE")
    parser.add_argument("--qe-alpha", type=float, default=3.0,
                        help="Alpha value for alpha-QE")
    parser.add_argument("--mmr", action="store_true",
                        help="Enable MMR diversification. Use only if validation improves.")
    parser.add_argument("--no-mmr", action="store_true",
                        help="Legacy flag; keeps MMR disabled")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not submit; save submission.json instead")

    parser.add_argument("--checkpoint", default=None,
                        help="Optional fine-tuned checkpoint. If absent, uses pretrained weights.")
    parser.add_argument("--arch", default="inception_resnet_v1",
                        choices=["inception_resnet_v1", "inception_resnet_v2"],
                        help="Backbone for pretrained mode. Checkpoint metadata overrides this.")

    parser.add_argument("--mmr-lambda", type=float, nargs="+", default=None,
                        help="MMR lambda schedule. One value is repeated for all ranks.")
    parser.add_argument("--k1", type=int, default=20,
                        help="k1 for k-reciprocal re-ranking")
    parser.add_argument("--k2", type=int, default=6,
                        help="k2 for k-reciprocal re-ranking")
    parser.add_argument("--kr-lambda", type=float, default=0.3,
                        help="k-reciprocal lambda value")
    parser.add_argument("--mmr-pool", type=int, default=50,
                        help="Initial candidate pool for MMR")

    parser.add_argument("--auto-crop", action="store_true",
                        help="Crop query/gallery with MTCNN before retrieval")
    parser.add_argument("--crop-image-size", type=int, default=None,
                        help="Crop size. Defaults to the encoder input size.")
    parser.add_argument("--crop-output-folder", default=".cropped_competition",
                        help="Where automatic crops are written")

    args = parser.parse_args()

    device = pick_device()
    print(f"[main] device = {device}")

    encoder = FaceNetEncoder(
        device=device,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
    )

    data_folder = Path(args.data_folder)
    if args.auto_crop:
        crop_size = args.crop_image_size or encoder.image_size
        data_folder = _prepare_auto_cropped_data(
            data_folder=data_folder,
            output_folder=Path(args.crop_output_folder),
            image_size=crop_size,
            device=device,
        )

    query_folder = data_folder / "query"
    gallery_folder = data_folder / "gallery"
    query_images, query_filenames = load_folder(query_folder)
    gallery_images, gallery_filenames = load_folder(gallery_folder)
    print(f"[main] query={len(query_images)} gallery={len(gallery_images)}")

    if not query_images:
        raise ValueError(f"No query images found in {query_folder}")
    if len(gallery_images) < 10:
        raise ValueError("The gallery must contain at least 10 images for submission.")

    use_mmr = bool(args.mmr and not args.no_mmr)

    system = RetrievalSystem(
        encoder=encoder,
        use_tta=not args.no_tta,
        use_kreciprocal=not args.no_kreciprocal,
        k1=args.k1,
        k2=args.k2,
        kr_lambda=args.kr_lambda,
        use_qe=args.qe,
        qe_top_k=args.qe_top_k,
        qe_alpha=args.qe_alpha,
        use_mmr=use_mmr,
        mmr_initial_pool=args.mmr_pool,
        mmr_lambda_schedule=_expand_mmr_schedule(args.mmr_lambda),
        top_k_output=10,
    )

    results = system.run(
        query_images, query_filenames,
        gallery_images, gallery_filenames,
    )

    for qfn, gfns in results.items():
        assert len(gfns) == 10, f"Query {qfn} has {len(gfns)} results"
        assert len(set(gfns)) == 10, f"Query {qfn} has duplicates: {gfns}"

    if args.dry_run:
        out_path = "submission.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"groupname": args.group_name, "images": results}, f, indent=2)
        print(f"[main] dry-run, saved to {out_path}")
    else:
        submit(results, args.group_name, args.submit_url)


if __name__ == "__main__":
    main()

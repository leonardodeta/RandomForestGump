"""
run_ablation.py
---------------
Run retrieval ablations on a local validation set.

Expected structure:
    val/
      query/
      gallery/
      ground_truth.json

``ground_truth.json`` format:
    {"query_filename.jpg": ["matching_gallery_1.jpg", "matching_gallery_2.jpg"]}
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from crop_faces import crop_folder
from facenet_encoder import FaceNetEncoder
from run_competition import load_folder, pick_device
from search_system import RetrievalSystem


def compute_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute Top-1, Top-5, Top-10 and weighted competition score."""
    top1 = top5 = top10 = 0
    evaluated = 0
    skipped = 0

    for qfn, pred in predictions.items():
        truth = set(ground_truth.get(qfn, []))
        if not truth:
            skipped += 1
            continue

        evaluated += 1
        if pred and pred[0] in truth:
            top1 += 1
        if any(p in truth for p in pred[:5]):
            top5 += 1
        if any(p in truth for p in pred[:10]):
            top10 += 1

    if evaluated == 0:
        raise ValueError(
            "No predictions had a valid ground-truth entry. Check filename keys in ground_truth.json."
        )

    top1_acc = top1 / evaluated
    top5_acc = top5 / evaluated
    top10_acc = top10 / evaluated
    score = 600 * top1_acc + 300 * top5_acc + 100 * top10_acc

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "top10": top10_acc,
        "score": score,
        "evaluated": evaluated,
        "skipped": skipped,
    }


def evaluate(system, query_folder, gallery_folder, ground_truth):
    qi, qfn = load_folder(query_folder)
    gi, gfn = load_folder(gallery_folder)
    preds = system.run(qi, qfn, gi, gfn, verbose=False)
    return compute_metrics(preds, ground_truth)


def _prepare_auto_cropped_val(val_folder: Path, output_folder: Path, image_size: int, device) -> Path:
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    crop_folder(val_folder / "query", output_folder / "query", image_size=image_size, device=device)
    crop_folder(val_folder / "gallery", output_folder / "gallery", image_size=image_size, device=device)
    shutil.copy2(val_folder / "ground_truth.json", output_folder / "ground_truth.json")
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-folder", required=True,
                        help="Validation folder with query/, gallery/, ground_truth.json")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional fine-tuned checkpoint. If absent, uses pretrained weights.")
    parser.add_argument("--arch", default="inception_resnet_v1",
                        choices=["inception_resnet_v1", "inception_resnet_v2"],
                        help="Backbone for pretrained mode. Checkpoint metadata overrides this.")
    parser.add_argument("--auto-crop", action="store_true",
                        help="Crop validation query/gallery before running ablations")
    parser.add_argument("--crop-image-size", type=int, default=None,
                        help="Crop size. Defaults to encoder input size.")
    parser.add_argument("--crop-output-folder", default=".cropped_ablation",
                        help="Where automatic validation crops are written")
    args = parser.parse_args()

    device = pick_device()
    print(f"[ablation] device = {device}")

    encoder = FaceNetEncoder(
        device=device,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
    )

    val_folder = Path(args.val_folder)
    if args.auto_crop:
        crop_size = args.crop_image_size or encoder.image_size
        val_folder = _prepare_auto_cropped_val(
            val_folder=val_folder,
            output_folder=Path(args.crop_output_folder),
            image_size=crop_size,
            device=device,
        )

    with open(val_folder / "ground_truth.json", encoding="utf-8") as f:
        ground_truth = json.load(f)

    query_folder = val_folder / "query"
    gallery_folder = val_folder / "gallery"

    configs = [
        ("Baseline cosine", dict(
            use_tta=False, use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("+ TTA", dict(
            use_tta=True, use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("+ k-reciprocal", dict(
            use_tta=True, use_kreciprocal=True, use_qe=False, use_mmr=False)),
        ("+ alpha-QE", dict(
            use_tta=True, use_kreciprocal=True, use_qe=True, use_mmr=False)),
        ("+ MMR diversif.", dict(
            use_tta=True, use_kreciprocal=True, use_qe=True, use_mmr=True)),
    ]

    print(f"\n{'Method':<28} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Score':>8} {'Eval':>6} {'Skip':>6}")
    print("-" * 84)
    for name, cfg in configs:
        system = RetrievalSystem(encoder=encoder, **cfg)
        metrics = evaluate(system, query_folder, gallery_folder, ground_truth)
        print(
            f"{name:<28} "
            f"{metrics['top1']:>7.3f} {metrics['top5']:>7.3f} "
            f"{metrics['top10']:>7.3f} {metrics['score']:>8.1f} "
            f"{metrics['evaluated']:>6d} {metrics['skipped']:>6d}"
        )


if __name__ == "__main__":
    main()

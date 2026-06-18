"""
run_ablation.py
---------------
Run validation ablations and parameter sweeps.

Expected structure:
    val/
      query/
      gallery/
      ground_truth.json

``ground_truth.json`` format:
    {"query_filename.jpg": ["matching_gallery_1.jpg", "matching_gallery_2.jpg"]}

The script prints a report table and also saves CSV/JSON files, so the numbers
can be copied directly into the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def compute_metrics(predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
    top1 = top5 = top10 = 0
    evaluated = 0
    empty_ground_truth = 0
    missing_predictions = 0

    for qfn, truth_values in ground_truth.items():
        truth = set(truth_values or [])
        if not truth:
            empty_ground_truth += 1
            continue

        evaluated += 1
        pred = predictions.get(qfn)
        if pred is None:
            missing_predictions += 1
            pred = []

        if pred and pred[0] in truth:
            top1 += 1
        if any(p in truth for p in pred[:5]):
            top5 += 1
        if any(p in truth for p in pred[:10]):
            top10 += 1

    if evaluated == 0:
        raise ValueError("No valid ground-truth entries found. Check ground_truth.json and filename keys.")

    extra_predictions = len(set(predictions) - set(ground_truth))
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
        "empty_ground_truth": empty_ground_truth,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
    }


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _prepare_auto_cropped_val(val_folder: Path, output_folder: Path, image_size: int, device) -> Path:
    from crop_faces import crop_folder

    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    crop_folder(val_folder / "query", output_folder / "query", image_size=image_size, device=device)
    crop_folder(val_folder / "gallery", output_folder / "gallery", image_size=image_size, device=device)
    shutil.copy2(val_folder / "ground_truth.json", output_folder / "ground_truth.json")
    return output_folder


def _base_configs() -> list[tuple[str, dict[str, object]]]:
    return [
        ("cosine", dict(use_tta=False, use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("tta", dict(use_tta=True, use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("tta+kreciprocal", dict(use_tta=True, use_kreciprocal=True, use_qe=False, use_mmr=False)),
        ("tta+qe", dict(use_tta=True, use_kreciprocal=False, use_qe=True, use_mmr=False)),
        ("tta+kreciprocal+qe", dict(use_tta=True, use_kreciprocal=True, use_qe=True, use_mmr=False)),
        ("tta+mmr", dict(use_tta=True, use_kreciprocal=False, use_qe=False, use_mmr=True)),
        ("tta+kreciprocal+qe+mmr", dict(use_tta=True, use_kreciprocal=True, use_qe=True, use_mmr=True)),
    ]


def _grid_configs(args) -> list[tuple[str, dict[str, object]]]:
    configs: list[tuple[str, dict[str, object]]] = []
    k1_values = _parse_int_list(args.k1_values)
    k2_values = _parse_int_list(args.k2_values)
    kr_values = _parse_float_list(args.kr_lambda_values)
    qe_top_values = _parse_int_list(args.qe_top_k_values)
    qe_alpha_values = _parse_float_list(args.qe_alpha_values)
    mmr_lambda_values = _parse_float_list(args.mmr_lambda_values)
    mmr_pool_values = _parse_int_list(args.mmr_pool_values)

    for k1 in k1_values:
        for k2 in k2_values:
            for kr in kr_values:
                configs.append((
                    f"grid:kreciprocal:k1={k1}:k2={k2}:lambda={kr}",
                    dict(use_tta=True, use_kreciprocal=True, k1=k1, k2=k2, kr_lambda=kr, use_qe=False, use_mmr=False),
                ))

    for top_k in qe_top_values:
        for alpha in qe_alpha_values:
            configs.append((
                f"grid:qe:top={top_k}:alpha={alpha}",
                dict(use_tta=True, use_kreciprocal=False, use_qe=True, qe_top_k=top_k, qe_alpha=alpha, use_mmr=False),
            ))

    for lam in mmr_lambda_values:
        for pool in mmr_pool_values:
            configs.append((
                f"grid:mmr:lambda={lam}:pool={pool}",
                dict(use_tta=True, use_kreciprocal=False, use_qe=False, use_mmr=True, mmr_lambda_schedule=[lam] * 10, mmr_initial_pool=pool),
            ))

    return configs


def _dedupe_configs(configs: Iterable[tuple[str, dict[str, object]]]) -> list[tuple[str, dict[str, object]]]:
    seen = set()
    out = []
    for name, cfg in configs:
        frozen = json.dumps(cfg, sort_keys=True, default=str)
        if frozen in seen:
            continue
        seen.add(frozen)
        out.append((name, cfg))
    return out


def _write_outputs(rows: list[dict[str, object]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "top1", "top5", "top10", "score", "evaluated", "missing_predictions",
        "extra_predictions", "use_tta", "use_kreciprocal", "k1", "k2", "kr_lambda",
        "use_qe", "qe_top_k", "qe_alpha", "use_mmr", "mmr_initial_pool", "mmr_lambda_schedule",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    from facenet_encoder import FaceNetEncoder
    from run_competition import load_folder, pick_device
    from search_system import RetrievalSystem

    parser = argparse.ArgumentParser()
    parser.add_argument("--val-folder", required=True, help="Validation folder with query/, gallery/, ground_truth.json")
    parser.add_argument("--checkpoint", default=None, help="Optional fine-tuned checkpoint")
    parser.add_argument("--arch", default="inception_resnet_v1", choices=["inception_resnet_v1", "inception_resnet_v2"])
    parser.add_argument("--auto-crop", action="store_true")
    parser.add_argument("--crop-image-size", type=int, default=None)
    parser.add_argument("--crop-output-folder", default=".cropped_ablation")
    parser.add_argument("--output-csv", default="ablation_results.csv")
    parser.add_argument("--output-json", default="ablation_results.json")
    parser.add_argument("--full-grid", action="store_true", help="Run parameter sweeps in addition to the compact ablation table")
    parser.add_argument("--k1-values", default="10,20,30")
    parser.add_argument("--k2-values", default="3,6,10")
    parser.add_argument("--kr-lambda-values", default="0.1,0.3,0.5")
    parser.add_argument("--qe-top-k-values", default="3,5,10")
    parser.add_argument("--qe-alpha-values", default="1.0,2.0,3.0")
    parser.add_argument("--mmr-lambda-values", default="0.5,0.7,0.9")
    parser.add_argument("--mmr-pool-values", default="20,50,100")
    parser.add_argument("--max-kreciprocal-matrix-elements", type=int, default=50_000_000)
    args = parser.parse_args()

    device = pick_device()
    print(f"[ablation] device = {device}")

    encoder = FaceNetEncoder(device=device, checkpoint_path=args.checkpoint, arch=args.arch)

    val_folder = Path(args.val_folder)
    if args.auto_crop:
        crop_size = args.crop_image_size or encoder.image_size
        val_folder = _prepare_auto_cropped_val(val_folder, Path(args.crop_output_folder), crop_size, device)

    with open(val_folder / "ground_truth.json", encoding="utf-8") as f:
        ground_truth = json.load(f)

    query_images, query_filenames = load_folder(val_folder / "query")
    gallery_images, gallery_filenames = load_folder(val_folder / "gallery")
    if not query_images:
        raise ValueError("Validation query folder is empty")
    if not gallery_images:
        raise ValueError("Validation gallery folder is empty")

    # Precompute embeddings once for no-TTA and once for TTA. All ranking-only
    # configurations reuse these features, which makes the ablation much faster.
    print("[ablation] Extracting baseline embeddings...")
    no_tta_embedder = RetrievalSystem(encoder=encoder, use_tta=False, use_kreciprocal=False)
    q_no_tta, g_no_tta = no_tta_embedder.embed_collections(query_images, gallery_images)

    print("[ablation] Extracting TTA embeddings...")
    tta_embedder = RetrievalSystem(encoder=encoder, use_tta=True, use_kreciprocal=False)
    q_tta, g_tta = tta_embedder.embed_collections(query_images, gallery_images)

    configs = _base_configs()
    if args.full_grid:
        configs.extend(_grid_configs(args))
    configs = _dedupe_configs(configs)

    rows: list[dict[str, object]] = []
    print(
        f"\n{'Method':<40} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} "
        f"{'Score':>8} {'Eval':>6} {'Miss':>6} {'Extra':>6}"
    )
    print("-" * 108)

    for name, cfg in configs:
        cfg = dict(cfg)
        cfg.setdefault("max_kreciprocal_matrix_elements", args.max_kreciprocal_matrix_elements)
        cfg.setdefault("top_k_output", 10)
        use_tta = bool(cfg.get("use_tta", False))
        q_feats, g_feats = (q_tta, g_tta) if use_tta else (q_no_tta, g_no_tta)

        system = RetrievalSystem(encoder=encoder, **cfg)
        try:
            predictions = system.run_from_embeddings(q_feats, query_filenames, g_feats, gallery_filenames)
            metrics = compute_metrics(predictions, ground_truth)
            error = ""
        except MemoryError as exc:
            metrics = {
                "top1": float("nan"), "top5": float("nan"), "top10": float("nan"), "score": float("nan"),
                "evaluated": 0, "missing_predictions": 0, "extra_predictions": 0,
            }
            error = str(exc)

        row = {"method": name, "error": error, **cfg, **metrics}
        rows.append(row)

        print(
            f"{name:<40} "
            f"{metrics['top1']:>7.3f} {metrics['top5']:>7.3f} "
            f"{metrics['top10']:>7.3f} {metrics['score']:>8.1f} "
            f"{int(metrics['evaluated']):>6d} {int(metrics['missing_predictions']):>6d} "
            f"{int(metrics['extra_predictions']):>6d}"
        )
        if error:
            print(f"  skipped: {error}")

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]) if r.get("score") == r.get("score") else float("-inf"), reverse=True)
    _write_outputs(rows_sorted, Path(args.output_csv), Path(args.output_json))
    print(f"\n[ablation] Saved CSV:  {args.output_csv}")
    print(f"[ablation] Saved JSON: {args.output_json}")
    if rows_sorted:
        best = rows_sorted[0]
        print(f"[ablation] Best method: {best['method']} | score={best['score']:.1f}")


if __name__ == "__main__":
    main()

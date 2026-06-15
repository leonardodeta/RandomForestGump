"""
run_ablation.py
---------------
Esegue tutti i componenti del search system in modalita' ablation sul
VALIDATION SET locale (quello che sta preparando il compagno).

Genera la tabella che andra' nel report nella sezione "Results and Ablations":

    Method                                | Top-1 | Top-5 | Top-10 | Score
    --------------------------------------|-------|-------|--------|------
    Baseline (cosine)                     |  ...  |  ...  |  ...   |  ...
    + TTA                                 |  ...  |  ...  |  ...   |  ...
    + k-reciprocal re-ranking             |  ...  |  ...  |  ...   |  ...
    + alpha-QE                            |  ...  |  ...  |  ...   |  ...
    + MMR diversification (full)          |  ...  |  ...  |  ...   |  ...

IMPORTANTE: lo script assume un validation set strutturato cosi':
    val/
      query/     <- immagini naturali
      gallery/   <- immagini sintetiche (inclusi distrattori)
      ground_truth.json   <- { query_filename: [lista filenames gallery con stessa identita'] }
"""

import os
import json
import argparse
from typing import Dict, List

from facenet_encoder import FaceNetEncoder
from search_system import RetrievalSystem
from run_competition import load_folder, pick_device


def compute_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Calcola Top-1, Top-5, Top-10 e final score (come la competizione)."""
    n = len(predictions)
    top1 = top5 = top10 = 0
    for qfn, pred in predictions.items():
        truth = set(ground_truth.get(qfn, []))
        if not truth:
            continue
        if pred[0] in truth:
            top1 += 1
        if any(p in truth for p in pred[:5]):
            top5 += 1
        if any(p in truth for p in pred[:10]):
            top10 += 1
    top1_acc = top1 / n
    top5_acc = top5 / n
    top10_acc = top10 / n
    score = 600 * top1_acc + 300 * top5_acc + 100 * top10_acc
    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "top10": top10_acc,
        "score": score,
    }


def evaluate(system, query_folder, gallery_folder, ground_truth):
    qi, qfn = load_folder(query_folder)
    gi, gfn = load_folder(gallery_folder)
    preds = system.run(qi, qfn, gi, gfn, verbose=False)
    return compute_metrics(preds, ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-folder", required=True,
                        help="Cartella di validazione con query/, gallery/, ground_truth.json")
    parser.add_argument("--checkpoint", default=None,
                        help="Path checkpoint SimCLR (opzionale). "
                             "Se non fornito usa pretrained VGGFace2.")
    args = parser.parse_args()

    device = pick_device()
    print(f"[ablation] device = {device}")

    with open(os.path.join(args.val_folder, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    query_folder = os.path.join(args.val_folder, "query")
    gallery_folder = os.path.join(args.val_folder, "gallery")

    # NB: lo stesso encoder viene riusato in tutte le configurazioni:
    # l'embedding e' lo stesso, cambia solo cosa fa il search system dopo.
    checkpoint = getattr(args, "checkpoint", None)
    encoder = FaceNetEncoder(device=device, checkpoint_path=checkpoint)

    configs = [
        ("Baseline (cosine)", dict(
            use_tta=False, use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("+ TTA",             dict(
            use_tta=True,  use_kreciprocal=False, use_qe=False, use_mmr=False)),
        ("+ k-reciprocal",    dict(
            use_tta=True,  use_kreciprocal=True,  use_qe=False, use_mmr=False)),
        ("+ alpha-QE",        dict(
            use_tta=True,  use_kreciprocal=True,  use_qe=True,  use_mmr=False)),
        ("+ MMR diversif.",   dict(
            use_tta=True,  use_kreciprocal=True,  use_qe=True,  use_mmr=True)),
    ]

    print(f"\n{'Method':<28} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Score':>8}")
    print("-" * 64)
    for name, cfg in configs:
        system = RetrievalSystem(encoder=encoder, **cfg)
        m = evaluate(system, query_folder, gallery_folder, ground_truth)
        print(f"{name:<28} {m['top1']:>7.3f} {m['top5']:>7.3f} "
              f"{m['top10']:>7.3f} {m['score']:>8.1f}")


if __name__ == "__main__":
    main()

"""
run_competition.py
------------------
Script da lanciare il GIORNO DELLA COMPETIZIONE.

Uso tipico:
    # Pretrained VGGFace2 (nessun checkpoint necessario):
    python run_competition.py \
        --data-folder /path/to/test_data \
        --group-name "NomeGruppo" \
        --submit-url http://localhost:3001/retrieval/

    # Con modello fine-tunato via SimCLR:
    python run_competition.py \
        --data-folder /path/to/test_data \
        --group-name "NomeGruppo" \
        --checkpoint simclr_checkpoint.pt \
        --submit-url http://localhost:3001/retrieval/
"""

import os
import json
import argparse
from typing import Tuple, List
import requests
import torch
from PIL import Image

from facenet_encoder import FaceNetEncoder
from search_system import RetrievalSystem


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_folder(folder: str) -> Tuple[List[Image.Image], List[str]]:
    """Carica tutte le immagini di una cartella mantenendo i nomi."""
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    images, filenames = [], []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(valid_ext):
            img = Image.open(os.path.join(folder, filename))
            images.append(img)
            filenames.append(filename)
    return images, filenames


def submit(results: dict, groupname: str, url: str) -> None:
    payload = json.dumps({"groupname": groupname, "images": results})
    response = requests.post(url, payload)
    try:
        out = json.loads(response.text)
        print(f"[submit] accuracy = {out.get('accuracy', out)}")
    except json.JSONDecodeError:
        print(f"[submit] ERROR: {response.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True,
                        help="Cartella test con sottocartelle query/ e gallery/")
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--submit-url",
                        default="http://localhost:3001/retrieval/")
    # Toggle componenti (per debug, ma di default tutto on tranne QE).
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-kreciprocal", action="store_true")
    parser.add_argument("--qe", action="store_true",
                        help="Attiva alpha-QE (default off, va testato sul val)")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Non chiamare la submit URL; salva su file.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path checkpoint SimCLR (opzionale). "
                             "Se non fornito usa pretrained VGGFace2.")
    args = parser.parse_args()

    device = pick_device()
    print(f"[main] device = {device}")

    # 1) Encoder: FaceNet pretrained su VGGFace2.
    #    Se disponibile un checkpoint SimCLR, passarlo con --checkpoint.
    checkpoint = getattr(args, "checkpoint", None)
    encoder = FaceNetEncoder(device=device, checkpoint_path=checkpoint)

    # 2) Carica query e gallery
    query_folder = os.path.join(args.data_folder, "query")
    gallery_folder = os.path.join(args.data_folder, "gallery")
    query_images, query_filenames = load_folder(query_folder)
    gallery_images, gallery_filenames = load_folder(gallery_folder)
    print(f"[main] query={len(query_images)} gallery={len(gallery_images)}")

    # 3) Sistema di retrieval con configurazione "default competition"
    system = RetrievalSystem(
        encoder=encoder,
        use_tta=not args.no_tta,
        use_kreciprocal=not args.no_kreciprocal,
        use_qe=args.qe,
        use_mmr=not args.no_mmr,
        top_k_output=10,
    )

    # 4) Run
    results = system.run(
        query_images, query_filenames,
        gallery_images, gallery_filenames,
    )

    # 5) Sanity check: ogni query deve avere ESATTAMENTE 10 gallery uniche
    for qfn, gfns in results.items():
        assert len(gfns) == 10, f"Query {qfn} ha {len(gfns)} risultati"
        assert len(set(gfns)) == 10, f"Query {qfn} ha duplicati: {gfns}"

    # 6) Submit (o salva)
    if args.dry_run:
        out_path = "submission.json"
        with open(out_path, "w") as f:
            json.dump({"groupname": args.group_name, "images": results}, f, indent=2)
        print(f"[main] dry-run, salvato in {out_path}")
    else:
        submit(results, args.group_name, args.submit_url)


if __name__ == "__main__":
    main()

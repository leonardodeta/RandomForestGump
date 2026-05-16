"""
crop_faces.py
-------------
Rileva e croppa i volti nelle immagini usando MTCNN.

Per ogni immagine nella cartella di input:
  - Se trova un volto, salva il crop (ridimensionato a --image-size)
  - Se NON trova un volto, salva l'immagine originale ridimensionata

Struttura attesa in input:
    Cartella piatta:
        query/img1.jpg, img2.jpg, ...
    Oppure struttura ImageFolder:
        query/classe_a/img1.jpg, ...

Uso tipico:
    python crop_faces.py --input query/ --output query_cropped/
    python crop_faces.py --input gallery/ --output gallery_cropped/ --image-size 160
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from facenet_pytorch import MTCNN
import torch


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(folder: Path):
    """Raccoglie tutti i path di immagini nella cartella (piatta o ImageFolder)."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def crop_and_save(
    img_path: Path,
    out_path: Path,
    mtcnn: MTCNN,
    image_size: int,
):
    """
    Croppa il volto da un'immagine e salva il risultato.
    Se non trova un volto, salva l'immagine originale ridimensionata.
    """
    img = Image.open(img_path).convert("RGB")

    # MTCNN ritorna il crop già ridimensionato a image_size se detect_face=True
    # Usiamo detect() per avere il bounding box e decidere cosa fare
    boxes, probs = mtcnn.detect(img)

    if boxes is not None and len(boxes) > 0:
        # Prendi il volto con confidenza più alta
        best_idx = probs.argmax()
        box = boxes[best_idx]  # [x1, y1, x2, y2]

        x1, y1, x2, y2 = [int(v) for v in box]

        # Clamp ai bordi dell'immagine
        w, h = img.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Piccolo padding (10%) per non tagliare troppo vicino al volto
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((image_size, image_size), Image.BILINEAR)
        face_found = True
    else:
        # Nessun volto trovato: ridimensiona l'immagine originale
        crop = img.resize((image_size, image_size), Image.BILINEAR)
        face_found = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return face_found


def main():
    parser = argparse.ArgumentParser(
        description="Croppa i volti nelle immagini usando MTCNN"
    )
    parser.add_argument("--input",      required=True, help="Cartella con le immagini originali")
    parser.add_argument("--output",     required=True, help="Cartella dove salvare i crop")
    parser.add_argument("--image-size", type=int, default=160, help="Dimensione output (default: 160)")
    parser.add_argument("--workers",    type=int, default=0,   help="Worker per MTCNN (default: 0)")
    parser.add_argument("--log-every",  type=int, default=100, help="Stampa ogni N immagini")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[crop_faces] device     = {device}")
    print(f"[crop_faces] input      = {args.input}")
    print(f"[crop_faces] output     = {args.output}")
    print(f"[crop_faces] image-size = {args.image_size}")

    mtcnn = MTCNN(
        image_size=args.image_size,
        margin=0,
        keep_all=False,
        device=device,
        post_process=False,
    )

    input_folder  = Path(args.input)
    output_folder = Path(args.output)
    img_paths     = get_image_paths(input_folder)

    print(f"[crop_faces] {len(img_paths)} immagini trovate")

    found    = 0
    not_found = 0

    for i, img_path in enumerate(img_paths):
        # Mantieni la struttura relativa delle sottocartelle
        rel_path = img_path.relative_to(input_folder)
        out_path = output_folder / rel_path

        try:
            face_found = crop_and_save(img_path, out_path, mtcnn, args.image_size)
            if face_found:
                found += 1
            else:
                not_found += 1
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")
            not_found += 1

        if (i + 1) % args.log_every == 0:
            print(f"  [{i+1}/{len(img_paths)}] volti trovati: {found} | non trovati: {not_found}")

    print(f"\n[crop_faces] Completato.")
    print(f"  Volti trovati:     {found} / {len(img_paths)}")
    print(f"  Non trovati (orig): {not_found} / {len(img_paths)}")
    print(f"  Output in: {output_folder}")


if __name__ == "__main__":
    main()

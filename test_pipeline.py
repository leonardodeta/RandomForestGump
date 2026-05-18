"""
test_pipeline.py
----------------
Smoke test: verifica che tutta la pipeline giri senza errori.
Non richiede dati reali — genera immagini casuali.

Uso:
    python test_pipeline.py
"""

import torch
from PIL import Image
import random

print("=" * 55)
print("SMOKE TEST PIPELINE")
print("=" * 55)

# --------------------------------------------------------
# 1. Import di tutti i moduli
# --------------------------------------------------------
print("\n[1/5] Import moduli...")
try:
    from encoder import Encoder
    from facenet_encoder import FaceNetEncoder
    from search_system import RetrievalSystem
    print("  OK")
except Exception as e:
    print(f"  ERRORE: {e}")
    raise

# --------------------------------------------------------
# 2. Device e encoder
# --------------------------------------------------------
print("\n[2/5] Caricamento FaceNetEncoder (pretrained VGGFace2)...")
try:
    device = torch.device("cpu")  # cpu per il test, non serve GPU
    encoder = FaceNetEncoder(device=device)
    print(f"  OK — embedding_dim = {encoder.embedding_dim}")
except Exception as e:
    print(f"  ERRORE: {e}")
    raise

# --------------------------------------------------------
# 3. Generazione immagini casuali
# --------------------------------------------------------
print("\n[3/5] Generazione immagini di test...")
N_QUERY   = 5
N_GALLERY = 20

def random_face_image():
    """Immagine RGB casuale 160x160 (simula un volto)."""
    return Image.fromarray(
        __import__("numpy").random.randint(0, 255, (160, 160, 3), dtype="uint8")
    )

query_images   = [random_face_image() for _ in range(N_QUERY)]
gallery_images = [random_face_image() for _ in range(N_GALLERY)]
query_names    = [f"query_{i:03d}.jpg"   for i in range(N_QUERY)]
gallery_names  = [f"gallery_{i:03d}.jpg" for i in range(N_GALLERY)]
print(f"  OK — {N_QUERY} query, {N_GALLERY} gallery")

# --------------------------------------------------------
# 4. embed_batch diretto
# --------------------------------------------------------
print("\n[4/5] Test embed_batch...")
try:
    feats = encoder.embed_batch(query_images[:2])
    assert feats.shape == (2, 512), f"Shape attesa (2, 512), ottenuta {feats.shape}"
    print(f"  OK — output shape: {feats.shape}")
except Exception as e:
    print(f"  ERRORE: {e}")
    raise

# --------------------------------------------------------
# 5. Pipeline completa con RetrievalSystem
# --------------------------------------------------------
print("\n[5/5] Test RetrievalSystem (cosine baseline, no re-ranking)...")
try:
    system = RetrievalSystem(
        encoder=encoder,
        use_tta=False,
        use_kreciprocal=False,
        use_qe=False,
        use_mmr=False,
        top_k_output=10,
    )
    results = system.run(
        query_images, query_names,
        gallery_images, gallery_names,
        verbose=False,
    )

    # Verifica formato output
    assert len(results) == N_QUERY, \
        f"Attese {N_QUERY} query nel dict, trovate {len(results)}"

    for qname, gnames in results.items():
        assert len(gnames) == 10, \
            f"Query {qname}: attesi 10 risultati, trovati {len(gnames)}"
        assert len(set(gnames)) == 10, \
            f"Query {qname}: risultati duplicati"

    print(f"  OK — {N_QUERY} query, ognuna con 10 gallery uniche")
    print(f"\n  Esempio: {query_names[0]} -> {results[query_names[0]][:3]}...")

except Exception as e:
    print(f"  ERRORE: {e}")
    raise

# --------------------------------------------------------
# Riepilogo
# --------------------------------------------------------
print("\n" + "=" * 55)
print("TUTTI I TEST PASSATI ✓")
print("La pipeline e' pronta.")
print("=" * 55)
print("""
Prossimi passi:
  Training SimCLR:
    python simclr_train.py --data-folder /path/to/dataset \\
                           --epochs 10 --batch-size 256

  Competition day:
    python run_competition.py --data-folder /path/to/test \\
                              --group-name "NomeGruppo" \\
                              --checkpoint simclr_checkpoint.pt
""")

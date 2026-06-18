"""
test_pipeline.py
----------------
Lightweight smoke tests for the retrieval pipeline.

Default mode uses a deterministic dummy encoder, so the test does not require
facenet-pytorch, pretrained weights, or internet access.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from PIL import Image

from encoder import Encoder
from reranking import alpha_query_expansion, k_reciprocal_rerank
from search_system import RetrievalSystem


class DummyEncoder(Encoder):
    def __init__(self, embedding_dim: int = 8):
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        feats = []
        for img in images:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            rgb = arr.mean(axis=(0, 1))
            base = torch.tensor([
                rgb[0], rgb[1], rgb[2],
                rgb.mean(), rgb.std(),
                rgb[0] - rgb[1], rgb[1] - rgb[2], rgb[2] - rgb[0],
            ], dtype=torch.float32)
            feats.append(base[: self.embedding_dim])
        return torch.stack(feats, dim=0) if feats else torch.empty((0, self.embedding_dim))


def make_image(seed: int, size: int = 32) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def run_lightweight_tests() -> None:
    print("=" * 60)
    print("LIGHTWEIGHT RETRIEVAL PIPELINE TEST")
    print("=" * 60)

    q = torch.nn.functional.normalize(torch.ones(2, 4), p=2, dim=1)
    g = torch.nn.functional.normalize(torch.ones(6, 4), p=2, dim=1)
    dist = k_reciprocal_rerank(q, g, k1=20, k2=6)
    assert dist.shape == (2, 6)
    assert torch.isfinite(dist).all(), "k-reciprocal produced NaN/Inf values"

    expanded = alpha_query_expansion(q, g[:2], top_k=5)
    assert expanded.shape == q.shape
    assert torch.isfinite(expanded).all(), "query expansion produced NaN/Inf values"
    print("[1/4] Utility functions OK")

    num_query = 5
    num_gallery = 20
    query_images = [make_image(i) for i in range(num_query)]
    gallery_images = [make_image(100 + i) for i in range(num_gallery)]
    query_names = [f"query_{i:03d}.jpg" for i in range(num_query)]
    gallery_names = [f"gallery_{i:03d}.jpg" for i in range(num_gallery)]

    default_system = RetrievalSystem(encoder=DummyEncoder())
    assert default_system.use_tta is True
    assert default_system.use_kreciprocal is False
    assert default_system.use_qe is False
    assert default_system.use_mmr is False
    print("[2/4] Conservative defaults OK")

    advanced = RetrievalSystem(
        encoder=DummyEncoder(),
        use_tta=True,
        use_kreciprocal=True,
        use_qe=True,
        use_mmr=True,
        top_k_output=10,
        mmr_initial_pool=5,
    )
    results = advanced.run(query_images, query_names, gallery_images, gallery_names, verbose=False)

    assert set(results) == set(query_names)
    for qname, ranked in results.items():
        assert len(ranked) == 10, f"{qname}: expected 10 results, got {len(ranked)}"
        assert len(set(ranked)) == 10, f"{qname}: duplicate gallery results: {ranked}"
        assert all(name in gallery_names for name in ranked)
    print("[3/4] Advanced RetrievalSystem output format OK")

    baseline = RetrievalSystem(
        encoder=DummyEncoder(),
        use_tta=False,
        use_kreciprocal=False,
        use_qe=False,
        use_mmr=False,
        top_k_output=10,
    )
    q_feats, g_feats = baseline.embed_collections(query_images, gallery_images)
    baseline_results = baseline.run_from_embeddings(q_feats, query_names, g_feats, gallery_names)
    assert all(len(v) == 10 for v in baseline_results.values())
    print("[4/4] Cached-embedding cosine path OK")

    print("\nAll lightweight tests passed.")


def run_facenet_integration_test() -> None:
    print("\nRunning optional FaceNet integration test on CPU...")
    from facenet_encoder import FaceNetEncoder

    encoder = FaceNetEncoder(device=torch.device("cpu"))
    images = [make_image(i, size=160) for i in range(2)]
    feats = encoder.embed_batch(images)
    assert feats.ndim == 2
    assert feats.shape[0] == 2
    assert feats.shape[1] == encoder.embedding_dim
    assert torch.isfinite(feats).all()
    print(f"FaceNet integration OK. Embedding shape: {tuple(feats.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-facenet", action="store_true", help="Also run a real FaceNetEncoder integration check")
    args = parser.parse_args()

    run_lightweight_tests()
    if args.with_facenet:
        run_facenet_integration_test()


if __name__ == "__main__":
    main()

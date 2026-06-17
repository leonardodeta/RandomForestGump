"""
search_system.py
----------------
Core retrieval pipeline.

The system receives an Encoder object and produces the competition submission
mapping query filenames to ranked gallery filenames. The baseline is deliberately
conservative: FaceNet embeddings + L2 normalization + cosine similarity. All
heavier post-processing components are explicit opt-ins so the final submission
can be justified by validation ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageOps

from encoder import Encoder
from reranking import alpha_query_expansion, k_reciprocal_rerank
from diversification import mmr_rerank


@dataclass(frozen=True)
class RetrievalConfig:
    query_batch_size: int = 32
    gallery_batch_size: int = 64
    use_tta: bool = True
    use_kreciprocal: bool = False
    k1: int = 20
    k2: int = 6
    kr_lambda: float = 0.3
    max_kreciprocal_matrix_elements: int = 50_000_000
    use_qe: bool = False
    qe_top_k: int = 5
    qe_alpha: float = 3.0
    use_mmr: bool = False
    mmr_lambda_schedule: Optional[List[float]] = None
    mmr_initial_pool: int = 50
    top_k_output: int = 10

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class RetrievalSystem:
    def __init__(
        self,
        encoder: Encoder,
        *,
        query_batch_size: int = 32,
        gallery_batch_size: int = 64,
        use_tta: bool = True,
        use_kreciprocal: bool = False,
        k1: int = 20,
        k2: int = 6,
        kr_lambda: float = 0.3,
        max_kreciprocal_matrix_elements: int = 50_000_000,
        use_qe: bool = False,
        qe_top_k: int = 5,
        qe_alpha: float = 3.0,
        use_mmr: bool = False,
        mmr_lambda_schedule: Optional[List[float]] = None,
        mmr_initial_pool: int = 50,
        top_k_output: int = 10,
    ):
        self.encoder = encoder
        self.config = RetrievalConfig(
            query_batch_size=query_batch_size,
            gallery_batch_size=gallery_batch_size,
            use_tta=use_tta,
            use_kreciprocal=use_kreciprocal,
            k1=k1,
            k2=k2,
            kr_lambda=kr_lambda,
            max_kreciprocal_matrix_elements=max_kreciprocal_matrix_elements,
            use_qe=use_qe,
            qe_top_k=qe_top_k,
            qe_alpha=qe_alpha,
            use_mmr=use_mmr,
            mmr_lambda_schedule=mmr_lambda_schedule,
            mmr_initial_pool=mmr_initial_pool,
            top_k_output=top_k_output,
        )
        self._validate_config()

    # ------------------------------------------------------------------ #
    # Compatibility properties for older code/tests.
    # ------------------------------------------------------------------ #
    @property
    def query_batch_size(self) -> int: return self.config.query_batch_size
    @property
    def gallery_batch_size(self) -> int: return self.config.gallery_batch_size
    @property
    def use_tta(self) -> bool: return self.config.use_tta
    @property
    def use_kreciprocal(self) -> bool: return self.config.use_kreciprocal
    @property
    def k1(self) -> int: return self.config.k1
    @property
    def k2(self) -> int: return self.config.k2
    @property
    def kr_lambda(self) -> float: return self.config.kr_lambda
    @property
    def use_qe(self) -> bool: return self.config.use_qe
    @property
    def qe_top_k(self) -> int: return self.config.qe_top_k
    @property
    def qe_alpha(self) -> float: return self.config.qe_alpha
    @property
    def use_mmr(self) -> bool: return self.config.use_mmr
    @property
    def mmr_lambda_schedule(self) -> Optional[List[float]]: return self.config.mmr_lambda_schedule
    @property
    def mmr_initial_pool(self) -> int: return self.config.mmr_initial_pool
    @property
    def top_k_output(self) -> int: return self.config.top_k_output

    def _validate_config(self) -> None:
        c = self.config
        if c.query_batch_size < 1 or c.gallery_batch_size < 1:
            raise ValueError("Batch sizes must be >= 1")
        if c.top_k_output < 1:
            raise ValueError("top_k_output must be >= 1")
        if c.k1 < 1:
            raise ValueError("k1 must be >= 1")
        if c.k2 < 1:
            raise ValueError("k2 must be >= 1")
        if not 0.0 <= c.kr_lambda <= 1.0:
            raise ValueError("kr_lambda must be in [0, 1]")
        if c.max_kreciprocal_matrix_elements < 1:
            raise ValueError("max_kreciprocal_matrix_elements must be >= 1")
        if c.qe_top_k < 1:
            raise ValueError("qe_top_k must be >= 1")
        if c.qe_alpha < 0:
            raise ValueError("qe_alpha must be non-negative")
        if c.mmr_initial_pool < 1:
            raise ValueError("mmr_initial_pool must be >= 1")
        if c.mmr_lambda_schedule is not None:
            for value in c.mmr_lambda_schedule:
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError("MMR lambda values must be in [0, 1]")

    # ------------------------------------------------------------------ #
    # Step 1: embedding extraction.
    # ------------------------------------------------------------------ #
    def _embed(self, images: List[Image.Image], batch_size: int) -> torch.Tensor:
        if not images:
            raise ValueError("Cannot embed an empty image list")

        feats = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            f = self.encoder.embed_batch(chunk)
            if f.ndim != 2:
                raise ValueError("encoder.embed_batch must return a 2-D tensor")
            if f.size(0) != len(chunk):
                raise ValueError("encoder returned a different number of embeddings than images")

            if self.use_tta:
                flipped = [ImageOps.mirror(img) for img in chunk]
                f_flip = self.encoder.embed_batch(flipped)
                if f_flip.shape != f.shape:
                    raise ValueError("TTA embeddings have a different shape from original embeddings")
                f = torch.nn.functional.normalize(f.float(), p=2, dim=1)
                f_flip = torch.nn.functional.normalize(f_flip.float(), p=2, dim=1)
                f = (f + f_flip) / 2.0
            feats.append(f.float())

        feats = torch.cat(feats, dim=0)
        return torch.nn.functional.normalize(feats, p=2, dim=1)

    def embed_collections(
        self,
        query_images: List[Image.Image],
        gallery_images: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._embed(query_images, self.query_batch_size),
            self._embed(gallery_images, self.gallery_batch_size),
        )

    # ------------------------------------------------------------------ #
    # Step 2: ranking.
    # ------------------------------------------------------------------ #
    def _rank(self, query_feats: torch.Tensor, gallery_feats: torch.Tensor) -> torch.Tensor:
        if query_feats.ndim != 2 or gallery_feats.ndim != 2:
            raise ValueError("query_feats and gallery_feats must be 2-D tensors")
        if query_feats.size(0) == 0 or gallery_feats.size(0) == 0:
            raise ValueError("query_feats and gallery_feats must be non-empty")
        if query_feats.size(1) != gallery_feats.size(1):
            raise ValueError("query_feats and gallery_feats must have the same embedding dimension")

        query_feats = torch.nn.functional.normalize(query_feats.float(), p=2, dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats.float(), p=2, dim=1)

        if self.use_qe:
            query_feats = alpha_query_expansion(
                query_feats,
                gallery_feats,
                top_k=self.qe_top_k,
                alpha=self.qe_alpha,
            )

        if self.use_kreciprocal:
            dist = k_reciprocal_rerank(
                query_feats,
                gallery_feats,
                k1=self.k1,
                k2=self.k2,
                lambda_value=self.kr_lambda,
                max_matrix_elements=self.config.max_kreciprocal_matrix_elements,
            )
            return -dist

        return query_feats @ gallery_feats.T

    # ------------------------------------------------------------------ #
    # Step 3: top-k selection.
    # ------------------------------------------------------------------ #
    def _select_top_k(self, score: torch.Tensor, gallery_feats: torch.Tensor) -> torch.Tensor:
        if score.ndim != 2:
            raise ValueError("score must have shape (num_queries, num_gallery)")
        if score.size(1) == 0:
            raise ValueError("score has zero gallery columns")
        k = min(self.top_k_output, score.size(1))
        if self.use_mmr:
            return mmr_rerank(
                score,
                gallery_feats,
                top_k=k,
                lambda_schedule=self.mmr_lambda_schedule,
                initial_pool=self.mmr_initial_pool,
            )
        _, idx = torch.topk(score, k=k, dim=1, largest=True, sorted=True)
        return idx

    def run_from_embeddings(
        self,
        query_feats: torch.Tensor,
        query_filenames: List[str],
        gallery_feats: torch.Tensor,
        gallery_filenames: List[str],
    ) -> Dict[str, List[str]]:
        if len(query_filenames) != query_feats.size(0):
            raise ValueError("Number of query filenames does not match query embeddings")
        if len(gallery_filenames) != gallery_feats.size(0):
            raise ValueError("Number of gallery filenames does not match gallery embeddings")

        score = self._rank(query_feats, gallery_feats)
        top_idx = self._select_top_k(score, gallery_feats)
        return {
            qfn: [gallery_filenames[j] for j in top_idx[i].tolist()]
            for i, qfn in enumerate(query_filenames)
        }

    # ------------------------------------------------------------------ #
    # Public API.
    # ------------------------------------------------------------------ #
    def run(
        self,
        query_images: List[Image.Image],
        query_filenames: List[str],
        gallery_images: List[Image.Image],
        gallery_filenames: List[str],
        verbose: bool = True,
    ) -> Dict[str, List[str]]:
        if len(query_images) != len(query_filenames):
            raise ValueError("query_images and query_filenames have different lengths")
        if len(gallery_images) != len(gallery_filenames):
            raise ValueError("gallery_images and gallery_filenames have different lengths")

        if verbose:
            print(f"[search] Embedding {len(query_images)} query images...")
        q_feats = self._embed(query_images, self.query_batch_size)

        if verbose:
            print(f"[search] Embedding {len(gallery_images)} gallery images...")
        g_feats = self._embed(gallery_images, self.gallery_batch_size)

        if verbose:
            print(
                "[search] Ranking "
                f"(tta={self.use_tta}, k-reciprocal={self.use_kreciprocal}, "
                f"qe={self.use_qe}, mmr={self.use_mmr})..."
            )
        return self.run_from_embeddings(q_feats, query_filenames, g_feats, gallery_filenames)

    def run_with_intermediates(
        self,
        query_images: List[Image.Image],
        query_filenames: List[str],
        gallery_images: List[Image.Image],
        gallery_filenames: List[str],
    ) -> Tuple[Dict[str, List[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        q_feats, g_feats = self.embed_collections(query_images, gallery_images)
        score = self._rank(q_feats, g_feats)
        top_idx = self._select_top_k(score, g_feats)
        results = {
            qfn: [gallery_filenames[j] for j in top_idx[i].tolist()]
            for i, qfn in enumerate(query_filenames)
        }
        return results, q_feats, g_feats, score

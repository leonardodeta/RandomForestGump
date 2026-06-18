"""
reranking.py
------------
Post-processing utilities for already-extracted face embeddings.

Implemented methods:
1. k-reciprocal re-ranking (Zhong et al., CVPR 2017), adapted for cosine
   distances on L2-normalized embeddings.
2. Alpha Query Expansion.

The functions are independent from FaceNet and are safe on small galleries:
parameters are validated and clamped where appropriate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


_EPS = 1e-12


def _validate_feature_matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (num_images, embedding_dim)")
    if value.size(0) == 0:
        raise ValueError(f"{name} must contain at least one embedding")


def k_reciprocal_rerank(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    max_matrix_elements: int = 50_000_000,
) -> torch.Tensor:
    """Apply k-reciprocal re-ranking.

    The method builds a dense ``(num_query + num_gallery)^2`` matrix. The
    ``max_matrix_elements`` guard prevents accidentally exhausting memory on a
    large hidden set. Disable the guard only if you know the machine has enough
    RAM.
    """
    _validate_feature_matrix("query_feats", query_feats)
    _validate_feature_matrix("gallery_feats", gallery_feats)
    if query_feats.size(1) != gallery_feats.size(1):
        raise ValueError("query_feats and gallery_feats must have the same embedding dimension")
    if not (0.0 <= lambda_value <= 1.0):
        raise ValueError("lambda_value must be in [0, 1]")
    if k1 < 1:
        raise ValueError("k1 must be >= 1")
    if k2 < 1:
        raise ValueError("k2 must be >= 1")
    if max_matrix_elements < 1:
        raise ValueError("max_matrix_elements must be >= 1")

    output_device = query_feats.device

    query_np = F.normalize(query_feats.detach().cpu().float(), p=2, dim=1).numpy().astype(np.float32)
    gallery_np = F.normalize(gallery_feats.detach().cpu().float(), p=2, dim=1).numpy().astype(np.float32)

    num_queries = query_np.shape[0]
    num_gallery = gallery_np.shape[0]
    total = num_queries + num_gallery
    matrix_elements = total * total
    if matrix_elements > max_matrix_elements:
        raise MemoryError(
            "k-reciprocal re-ranking would allocate a dense "
            f"{total}x{total} matrix ({matrix_elements:,} elements), above the "
            f"configured guard of {max_matrix_elements:,}. Use cosine ranking, "
            "reduce the validation set, or increase --max-kreciprocal-matrix-elements."
        )

    all_feats = np.concatenate([query_np, gallery_np], axis=0)

    # The +1 convention below includes the item itself among nearest neighbours.
    k1 = min(int(k1), max(total - 1, 1))
    k2 = min(int(k2), total)

    original_dist = 1.0 - all_feats @ all_feats.T
    original_dist = np.clip(original_dist, 0.0, 2.0).astype(np.float32)

    denom = np.max(original_dist, axis=0, keepdims=True)
    original_dist = original_dist / np.maximum(denom, _EPS)
    original_dist = np.nan_to_num(original_dist, nan=0.0, posinf=1.0, neginf=0.0)

    initial_rank = np.argsort(original_dist, axis=1)
    V = np.zeros((total, total), dtype=np.float32)

    for i in range(total):
        forward_k_neigh = initial_rank[i, : k1 + 1]
        backward_k_neigh = initial_rank[forward_k_neigh, : k1 + 1]
        reciprocal_positions = np.where(backward_k_neigh == i)[0]
        k_reciprocal_index = forward_k_neigh[reciprocal_positions]

        if k_reciprocal_index.size == 0:
            k_reciprocal_index = np.array([i], dtype=np.int64)

        k_reciprocal_expansion = k_reciprocal_index.copy()
        half_k1 = max(1, int(np.around(k1 / 2.0)))

        for candidate in k_reciprocal_index:
            cand_forward = initial_rank[candidate, : half_k1 + 1]
            cand_backward = initial_rank[cand_forward, : half_k1 + 1]
            cand_positions = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_forward[cand_positions]
            if cand_reciprocal.size == 0:
                continue
            overlap = len(np.intersect1d(cand_reciprocal, k_reciprocal_index))
            if overlap > (2.0 / 3.0) * len(cand_reciprocal):
                k_reciprocal_expansion = np.append(k_reciprocal_expansion, cand_reciprocal)

        k_reciprocal_expansion = np.unique(k_reciprocal_expansion)
        weights = np.exp(-original_dist[i, k_reciprocal_expansion]).astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= _EPS or not np.isfinite(weight_sum):
            V[i, i] = 1.0
        else:
            V[i, k_reciprocal_expansion] = weights / weight_sum

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(total):
            V_qe[i] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(total)]
    jaccard_dist = np.zeros((num_queries, total), dtype=np.float32)

    for i in range(num_queries):
        temp_min = np.zeros(total, dtype=np.float32)
        non_zero = np.where(V[i, :] != 0)[0]
        for ind in non_zero:
            related_images = inv_index[ind]
            temp_min[related_images] += np.minimum(V[i, ind], V[related_images, ind])
        jaccard_dist[i] = 1.0 - temp_min / np.maximum(2.0 - temp_min, _EPS)

    final_dist = (
        jaccard_dist * (1.0 - lambda_value)
        + original_dist[:num_queries, :] * lambda_value
    )
    final_dist = final_dist[:, num_queries:]
    final_dist = np.clip(final_dist, 0.0, None)
    final_dist = np.nan_to_num(final_dist, nan=0.0, posinf=2.0, neginf=0.0)

    return torch.from_numpy(final_dist).to(output_device)


def alpha_query_expansion(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    top_k: int = 5,
    alpha: float = 3.0,
) -> torch.Tensor:
    """Apply alpha query expansion to query embeddings."""
    _validate_feature_matrix("query_feats", query_feats)
    _validate_feature_matrix("gallery_feats", gallery_feats)
    if query_feats.size(1) != gallery_feats.size(1):
        raise ValueError("query_feats and gallery_feats must have the same embedding dimension")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    query_feats = F.normalize(query_feats.float(), p=2, dim=1)
    gallery_feats = F.normalize(gallery_feats.float(), p=2, dim=1)

    safe_top_k = min(int(top_k), gallery_feats.size(0))
    sim = query_feats @ gallery_feats.T
    top_sims, top_idx = torch.topk(sim, k=safe_top_k, dim=1)

    weights = torch.clamp(top_sims, min=0.0) ** alpha
    weight_sums = weights.sum(dim=1, keepdim=True)

    uniform = torch.full_like(weights, 1.0 / safe_top_k)
    weights = torch.where(weight_sums > _EPS, weights / (weight_sums + _EPS), uniform)

    expanded = (weights.unsqueeze(-1) * gallery_feats[top_idx]).sum(dim=1)
    return F.normalize(query_feats + expanded, p=2, dim=1)

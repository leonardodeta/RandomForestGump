"""
diversification.py
------------------
Optional MMR diversification for the final retrieval ranking.

For face retrieval, diversity is not always desirable: the metric usually
rewards returning many images of the same identity. Therefore MMR is disabled by
default in the main pipeline and should only be enabled when validation ablation
shows a real gain.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def _default_lambda_schedule(top_k: int) -> list[float]:
    base = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    if top_k <= len(base):
        return base[:top_k]
    return base + [base[-1]] * (top_k - len(base))


def mmr_rerank(
    query_sim_to_gallery: torch.Tensor,
    gallery_feats: torch.Tensor,
    top_k: int = 10,
    lambda_schedule: Optional[Iterable[float]] = None,
    initial_pool: int = 50,
) -> torch.Tensor:
    """
    Apply Maximal Marginal Relevance independently for each query.

    MMR score at a given rank is:
        lambda * sim(query, candidate)
        - (1 - lambda) * max_sim(candidate, already_selected)

    Args:
        query_sim_to_gallery: Tensor of shape ``(M, N)`` where larger values are
            better query-gallery matches.
        gallery_feats: Tensor of shape ``(N, D)``. The function normalizes these
            features defensively before computing candidate-candidate similarity.
        top_k: Number of gallery items to return per query.
        lambda_schedule: Optional per-rank lambda values. A single value should
            be expanded by the caller, but this function also accepts any list
            with at least ``top_k`` values.
        initial_pool: Candidate pool size per query. It is automatically clamped
            to ``[top_k, N]`` so that MMR cannot produce duplicate results.

    Returns:
        LongTensor of shape ``(M, min(top_k, N))`` containing selected gallery
        indices in ranked order.
    """
    if query_sim_to_gallery.ndim != 2:
        raise ValueError("query_sim_to_gallery must have shape (num_queries, num_gallery)")
    if gallery_feats.ndim != 2:
        raise ValueError("gallery_feats must have shape (num_gallery, embedding_dim)")
    if query_sim_to_gallery.size(1) != gallery_feats.size(0):
        raise ValueError("gallery_feats must contain one feature per gallery column")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if initial_pool < 1:
        raise ValueError("initial_pool must be >= 1")

    num_queries, num_gallery = query_sim_to_gallery.shape
    safe_top_k = min(int(top_k), num_gallery)
    safe_pool = max(safe_top_k, min(int(initial_pool), num_gallery))

    if lambda_schedule is None:
        lambdas = _default_lambda_schedule(safe_top_k)
    else:
        lambdas = list(lambda_schedule)
        if len(lambdas) == 1:
            lambdas = lambdas * safe_top_k
        if len(lambdas) < safe_top_k:
            raise ValueError("lambda_schedule must contain at least top_k values")
        lambdas = lambdas[:safe_top_k]

    for value in lambdas:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("MMR lambda values must be in [0, 1]")

    device = query_sim_to_gallery.device
    gallery_feats = F.normalize(gallery_feats.to(device).float(), p=2, dim=1)
    scores = query_sim_to_gallery.float()

    pool_sims, pool_idx = torch.topk(scores, k=safe_pool, dim=1)
    selected_idx = torch.empty((num_queries, safe_top_k), dtype=torch.long, device=device)

    for q in range(num_queries):
        cand_idx = pool_idx[q]
        cand_sims = pool_sims[q]
        cand_feats = gallery_feats[cand_idx]
        inter_sim = cand_feats @ cand_feats.T

        chosen_positions: list[int] = []
        chosen_mask = torch.zeros(safe_pool, dtype=torch.bool, device=device)

        for rank in range(safe_top_k):
            lam = float(lambdas[rank])
            if rank == 0:
                rank_scores = cand_sims.clone()
            else:
                max_inter = inter_sim[:, chosen_positions].max(dim=1).values
                rank_scores = lam * cand_sims - (1.0 - lam) * max_inter
            rank_scores[chosen_mask] = float("-inf")
            best = int(torch.argmax(rank_scores).item())
            chosen_positions.append(best)
            chosen_mask[best] = True

        selected_idx[q] = cand_idx[torch.tensor(chosen_positions, device=device)]

    return selected_idx

"""
reranking.py
------------
Re-ranking degli embedding gia' estratti.

Implementa:
1. k-reciprocal re-ranking (Zhong et al., CVPR 2017)
2. Query Expansion (alpha-QE)

Entrambi lavorano SOLO sugli embedding: non sanno nulla del modello
sottostante. Questo li rende riutilizzabili e testabili in isolamento.
"""

import torch
import numpy as np


def k_reciprocal_rerank(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> torch.Tensor:
    """
    k-reciprocal re-ranking.

    Idea: due immagini A e B sono *davvero* simili se A e' nei top-k vicini
    di B E B e' nei top-k vicini di A. Si calcola una distanza di Jaccard
    sui set di k-vicini reciproci e la si combina con la cosine originale.

    Args:
        query_feats: (M, D) embedding query L2-normalizzati
        gallery_feats: (N, D) embedding gallery L2-normalizzati
        k1: numero di vicini per costruire l'insieme k-reciproco (tipico 20)
        k2: numero di vicini per la query expansion locale (tipico 6)
        lambda_value: peso della distanza originale vs Jaccard.
                      0 = solo Jaccard, 1 = solo cosine. Tipico 0.3.

    Returns:
        Matrice di distanze finali (M, N). Valori PICCOLI = match migliore.
        (Attenzione: e' distanza, non similarita'. Per ottenere ranking:
        usare torch.topk con largest=False, oppure -final_dist con largest=True.)
    """
    # Lavoriamo in numpy per coerenza con l'implementazione di riferimento
    # di Zhong et al. che e' lo standard nel re-ID.
    query_feats = query_feats.cpu().numpy().astype(np.float32)
    gallery_feats = gallery_feats.cpu().numpy().astype(np.float32)

    M = query_feats.shape[0]
    N = gallery_feats.shape[0]

    # Concateno query e gallery in un unico insieme: serve calcolare le
    # distanze reciproche tra TUTTI gli elementi (query incluse).
    all_feats = np.concatenate([query_feats, gallery_feats], axis=0)
    total = M + N

    # Distanza coseno = 1 - similarita coseno (feature gia' normalizzate).
    original_dist = 1.0 - all_feats @ all_feats.T
    original_dist = np.clip(original_dist, 0.0, 2.0)

    # Normalizzo ogni colonna in [0,1] (passo standard di Zhong et al.).
    original_dist = original_dist / np.max(original_dist, axis=0, keepdims=True)

    # === Costruzione dell'insieme k-reciproco esteso per ciascun elemento ===
    V = np.zeros((total, total), dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1)

    for i in range(total):
        # Top k1+1 vicini di i (escludendo se stesso lo daremo per scontato)
        forward_k_neigh = initial_rank[i, : k1 + 1]
        # Backward: per ogni j tra i forward, controllo che i sia nei suoi top-k1
        backward_k_neigh = initial_rank[forward_k_neigh, : k1 + 1]
        fi = np.where(backward_k_neigh == i)[0]
        k_reciprocal_index = forward_k_neigh[fi]

        # Estensione: aggiungi vicini "quasi reciproci" (vedi paper).
        k_reciprocal_expansion = k_reciprocal_index.copy()
        for candidate in k_reciprocal_index:
            half_k1 = int(np.around(k1 / 2.0))
            cand_forward = initial_rank[candidate, : half_k1 + 1]
            cand_backward = initial_rank[cand_forward, : half_k1 + 1]
            fi_cand = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_forward[fi_cand]
            # Aggiungi se intersezione abbastanza grande
            if (
                len(np.intersect1d(cand_reciprocal, k_reciprocal_index))
                > 2.0 / 3.0 * len(cand_reciprocal)
            ):
                k_reciprocal_expansion = np.append(
                    k_reciprocal_expansion, cand_reciprocal
                )

        k_reciprocal_expansion = np.unique(k_reciprocal_expansion)

        # Pesi gaussiani sulla distanza originale
        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)

    # === Local query expansion: media i vettori V dei top-k2 ===
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(total):
            V_qe[i] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    # === Distanza di Jaccard tra i set k-reciproci ===
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(total)]
    jaccard_dist = np.zeros((M, total), dtype=np.float32)

    for i in range(M):
        temp_min = np.zeros(total, dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[indImages[j]] += np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    # === Combinazione finale ===
    final_dist = (
        jaccard_dist * (1.0 - lambda_value)
        + original_dist[:M, :] * lambda_value
    )
    # Restituisco solo le distanze query -> gallery
    final_dist = final_dist[:, M:]
    final_dist = np.clip(final_dist, 0.0, None)

    return torch.from_numpy(final_dist)


def alpha_query_expansion(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    top_k: int = 5,
    alpha: float = 3.0,
) -> torch.Tensor:
    """
    Alpha Query Expansion (Radenovic et al., 2018).

    Per ogni query, calcola un nuovo embedding come media pesata della query
    stessa e dei suoi top-k match in gallery, con pesi sim^alpha.
    Poi ri-normalizza. Le query "si arricchiscono" delle loro match migliori.

    Args:
        query_feats: (M, D) L2-normalizzati
        gallery_feats: (N, D) L2-normalizzati
        top_k: quanti gallery usare per espandere ogni query
        alpha: esponente dei pesi (piu' alto = piu' selettivo)

    Returns:
        (M, D) query expanded, L2-normalizzate.
    """
    sim = query_feats @ gallery_feats.T            # (M, N)
    top_sims, top_idx = torch.topk(sim, k=top_k, dim=1)  # (M, k)

    # Pesi: sim^alpha (con sim clampato >= 0 per evitare segni assurdi)
    weights = torch.clamp(top_sims, min=0.0) ** alpha    # (M, k)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    # Embedding gallery dei top-k per ciascuna query
    expanded = (weights.unsqueeze(-1) * gallery_feats[top_idx]).sum(dim=1)

    # Combina query originale + espansione, poi normalizza
    new_query = query_feats + expanded
    new_query = torch.nn.functional.normalize(new_query, p=2, dim=1)
    return new_query

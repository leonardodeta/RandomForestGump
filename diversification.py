"""
diversification.py
------------------
Diversificazione controllata del ranking finale (rank 2-10).

Idea (vedi conversazione col gruppo):
- Rank 1 = sempre il match piu' forte. Niente filtri.
- Rank 2-N = applica MMR (Maximal Marginal Relevance) per evitare
  che i risultati siano "10 copie quasi identiche" della stessa immagine.

ATTENZIONE: la metrica della competizione premia avere la persona giusta,
NON la diversita'. Quindi MMR e' un *hedge*: se sei sicuro non cambia
quasi nulla, se sei incerto copre piu' ipotesi. Va validato su val set.
"""

import torch


def mmr_rerank(
    query_sim_to_gallery: torch.Tensor,
    gallery_feats: torch.Tensor,
    top_k: int = 10,
    lambda_schedule=None,
    initial_pool: int = 50,
) -> torch.Tensor:
    """
    Applica MMR a ciascuna query indipendentemente.

    A ogni step di ranking, sceglie la gallery image che massimizza:
        score(g) = lambda * sim(query, g) - (1-lambda) * max sim(g, g_selected)

    Args:
        query_sim_to_gallery: (M, N) similarita' query->gallery (es. cosine
            o -final_dist da k-reciprocal). Valori GRANDI = match migliore.
        gallery_feats: (N, D) embedding gallery L2-normalizzati, servono
            per calcolare la similarita' inter-gallery.
        top_k: quanti elementi selezionare per query (10 per la competizione).
        lambda_schedule: lista di top_k valori lambda, uno per ogni rank.
            Default: 1.0 per il rank 1, poi scala decrescente da 0.9 a 0.5.
            lambda=1 -> solo rilevanza; lambda=0 -> solo diversita'.
        initial_pool: candidati iniziali per query (top-initial_pool per
            cosine). Riduce il costo computazionale: senza questo, MMR
            dovrebbe considerare TUTTA la gallery a ogni step.

    Returns:
        (M, top_k) indici gallery selezionati per ogni query, in ordine.
    """
    M, N = query_sim_to_gallery.shape
    initial_pool = min(initial_pool, N)

    if lambda_schedule is None:
        # Rank 1: solo rilevanza. Rank 2-5: poca diversita'.
        # Rank 6-10: piu' diversita' (hedging sul top-10).
        lambda_schedule = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    assert len(lambda_schedule) >= top_k

    # Pre-calcolo: per ogni query, i candidati iniziali (top-initial_pool).
    pool_sims, pool_idx = torch.topk(query_sim_to_gallery, k=initial_pool, dim=1)
    # pool_idx: (M, P) — indici gallery candidati per ogni query
    # pool_sims: (M, P) — sim originale

    # Similarita' inter-gallery sui soli candidati (per ogni query): (M, P, P)
    # Per evitare blow-up di memoria con M grande, lavoriamo per-query.

    selected_idx = torch.zeros((M, top_k), dtype=torch.long)

    for q in range(M):
        cand_idx = pool_idx[q]                       # (P,)
        cand_sims = pool_sims[q]                     # (P,) sim alla query
        cand_feats = gallery_feats[cand_idx]         # (P, D)
        # Similarita' interna tra candidati: (P, P)
        inter_sim = cand_feats @ cand_feats.T

        P = cand_idx.shape[0]
        chosen = []                                  # posizioni nel pool
        chosen_mask = torch.zeros(P, dtype=torch.bool)

        for rank in range(top_k):
            lam = lambda_schedule[rank]
            if rank == 0:
                # Rank 1: il piu' simile alla query, senza penalita'.
                best = int(torch.argmax(cand_sims).item())
            else:
                # max sim verso elementi gia' scelti
                max_inter = inter_sim[:, chosen].max(dim=1).values  # (P,)
                mmr_score = lam * cand_sims - (1.0 - lam) * max_inter
                mmr_score[chosen_mask] = float("-inf")              # no duplicati
                best = int(torch.argmax(mmr_score).item())

            chosen.append(best)
            chosen_mask[best] = True

        selected_idx[q] = cand_idx[torch.tensor(chosen)]

    return selected_idx

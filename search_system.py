"""
search_system.py
----------------
Il cuore del sistema di retrieval.

Riceve un Encoder (qualsiasi modello che rispetti l'interfaccia) e fornisce
il metodo `run` che dato query e gallery produce il dict di submission.

Ogni componente e' OPZIONALE e configurabile dal costruttore. Questo permette
di fare gli ablation per il report semplicemente cambiando i flag:
    - use_tta
    - use_kreciprocal
    - use_qe
    - use_mmr
"""

from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image, ImageOps

from encoder import Encoder
from reranking import k_reciprocal_rerank, alpha_query_expansion
from diversification import mmr_rerank


class RetrievalSystem:
    def __init__(
        self,
        encoder: Encoder,
        *,
        # --- batching ---
        query_batch_size: int = 32,
        gallery_batch_size: int = 64,
        # --- Test-Time Augmentation ---
        use_tta: bool = True,
        # --- k-reciprocal re-ranking ---
        use_kreciprocal: bool = True,
        k1: int = 20,
        k2: int = 6,
        kr_lambda: float = 0.3,
        # --- alpha-QE ---
        use_qe: bool = False,
        qe_top_k: int = 5,
        qe_alpha: float = 3.0,
        # --- MMR diversification ---
        # Disabled by default: for identity retrieval, diversity can remove
        # correct same-identity images unless validation proves otherwise.
        use_mmr: bool = False,
        mmr_lambda_schedule: Optional[List[float]] = None,
        mmr_initial_pool: int = 50,
        # --- output ---
        top_k_output: int = 10,
    ):
        self.encoder = encoder
        self.query_batch_size = query_batch_size
        self.gallery_batch_size = gallery_batch_size

        self.use_tta = use_tta
        self.use_kreciprocal = use_kreciprocal
        self.k1 = k1
        self.k2 = k2
        self.kr_lambda = kr_lambda

        self.use_qe = use_qe
        self.qe_top_k = qe_top_k
        self.qe_alpha = qe_alpha

        self.use_mmr = use_mmr
        self.mmr_lambda_schedule = mmr_lambda_schedule
        self.mmr_initial_pool = mmr_initial_pool

        self.top_k_output = top_k_output

    # ------------------------------------------------------------------ #
    # Step 1: estrazione embedding (con TTA opzionale)
    # ------------------------------------------------------------------ #
    def _embed(
        self, images: List[Image.Image], batch_size: int
    ) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            f = self.encoder.embed_batch(chunk)
            if self.use_tta:
                # Flip orizzontale: media degli embedding dopo normalizzazione.
                # (Per i volti il flip e' quasi sempre safe; se per qualche motivo
                # le query hanno orientamento specifico, disattivare TTA.)
                flipped = [ImageOps.mirror(img) for img in chunk]
                f_flip = self.encoder.embed_batch(flipped)
                f = torch.nn.functional.normalize(f, p=2, dim=1)
                f_flip = torch.nn.functional.normalize(f_flip, p=2, dim=1)
                f = (f + f_flip) / 2.0
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        # Normalizzazione finale (necessaria comunque, anche senza TTA)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats

    # ------------------------------------------------------------------ #
    # Step 2: calcolo del ranking
    # ------------------------------------------------------------------ #
    def _rank(
        self,
        query_feats: torch.Tensor,
        gallery_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Restituisce una matrice (M, N) di SCORE: valori grandi = match migliore.
        """
        # Optional: query expansion sulla query feats (prima del re-ranking)
        if self.use_qe:
            query_feats = alpha_query_expansion(
                query_feats, gallery_feats,
                top_k=self.qe_top_k, alpha=self.qe_alpha,
            )

        if self.use_kreciprocal:
            # k-reciprocal restituisce DISTANZE (piccolo = meglio).
            # Convertiamo in score negandolo.
            dist = k_reciprocal_rerank(
                query_feats, gallery_feats,
                k1=self.k1, k2=self.k2, lambda_value=self.kr_lambda,
            )
            score = -dist
        else:
            # Cosine similarity diretta (feature gia' normalizzate)
            score = query_feats @ gallery_feats.T
        return score

    # ------------------------------------------------------------------ #
    # Step 3: selezione top-k (con MMR opzionale) e costruzione submission
    # ------------------------------------------------------------------ #
    def _select_top_k(
        self,
        score: torch.Tensor,
        gallery_feats: torch.Tensor,
    ) -> torch.Tensor:
        k = min(self.top_k_output, score.size(1))
        if self.use_mmr:
            return mmr_rerank(
                score, gallery_feats,
                top_k=k,
                lambda_schedule=self.mmr_lambda_schedule,
                initial_pool=self.mmr_initial_pool,
            )
        _, idx = torch.topk(score, k=k, dim=1)
        return idx

    # ------------------------------------------------------------------ #
    # API pubblica
    # ------------------------------------------------------------------ #
    def run(
        self,
        query_images: List[Image.Image],
        query_filenames: List[str],
        gallery_images: List[Image.Image],
        gallery_filenames: List[str],
        verbose: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Esegue l'intera pipeline di retrieval.

        Returns:
            Dict { query_filename -> [10 gallery_filenames ordinati] }
            nel formato richiesto dalla submission.
        """
        if verbose:
            print(f"[search] Embedding {len(query_images)} query...")
        q_feats = self._embed(query_images, self.query_batch_size)

        if verbose:
            print(f"[search] Embedding {len(gallery_images)} gallery...")
        g_feats = self._embed(gallery_images, self.gallery_batch_size)

        if verbose:
            print(
                f"[search] Ranking "
                f"(re-rank={self.use_kreciprocal}, qe={self.use_qe})..."
            )
        score = self._rank(q_feats, g_feats)

        if verbose:
            print(f"[search] Selecting top-{self.top_k_output} (mmr={self.use_mmr})...")
        top_idx = self._select_top_k(score, g_feats)

        # Costruzione dizionario di output
        results: Dict[str, List[str]] = {}
        for i, qfn in enumerate(query_filenames):
            results[qfn] = [gallery_filenames[j] for j in top_idx[i].tolist()]
        return results

    # ------------------------------------------------------------------ #
    # Utility: ritorna anche embedding e score, utile per analisi/ablation
    # ------------------------------------------------------------------ #
    def run_with_intermediates(
        self,
        query_images: List[Image.Image],
        query_filenames: List[str],
        gallery_images: List[Image.Image],
        gallery_filenames: List[str],
    ) -> Tuple[Dict[str, List[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        q_feats = self._embed(query_images, self.query_batch_size)
        g_feats = self._embed(gallery_images, self.gallery_batch_size)
        score = self._rank(q_feats, g_feats)
        top_idx = self._select_top_k(score, g_feats)
        results = {
            qfn: [gallery_filenames[j] for j in top_idx[i].tolist()]
            for i, qfn in enumerate(query_filenames)
        }
        return results, q_feats, g_feats, score

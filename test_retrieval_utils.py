"""
test_retrieval_utils.py
-----------------------
Unit-style tests for retrieval utilities that do not require facenet-pytorch.

Run with either:
    python test_retrieval_utils.py

or, if pytest is available:
    pytest test_retrieval_utils.py
"""

from __future__ import annotations

import unittest

import torch

from diversification import mmr_rerank
from reranking import alpha_query_expansion, k_reciprocal_rerank
from run_ablation import compute_metrics


class RetrievalUtilityTests(unittest.TestCase):
    def test_k_reciprocal_handles_identical_embeddings_without_nan(self):
        query = torch.nn.functional.normalize(torch.ones(2, 4), p=2, dim=1)
        gallery = torch.nn.functional.normalize(torch.ones(5, 4), p=2, dim=1)
        dist = k_reciprocal_rerank(query, gallery, k1=20, k2=6)
        self.assertEqual(tuple(dist.shape), (2, 5))
        self.assertTrue(torch.isfinite(dist).all().item())

    def test_alpha_query_expansion_clamps_top_k_to_gallery_size(self):
        query = torch.nn.functional.normalize(torch.randn(3, 8), p=2, dim=1)
        gallery = torch.nn.functional.normalize(torch.randn(2, 8), p=2, dim=1)
        expanded = alpha_query_expansion(query, gallery, top_k=10, alpha=3.0)
        self.assertEqual(tuple(expanded.shape), tuple(query.shape))
        norms = expanded.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_mmr_pool_smaller_than_top_k_still_returns_unique_results(self):
        scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
        gallery = torch.nn.functional.normalize(torch.randn(6, 4), p=2, dim=1)
        idx = mmr_rerank(scores, gallery, top_k=5, initial_pool=2)
        self.assertEqual(tuple(idx.shape), (1, 5))
        values = idx[0].tolist()
        self.assertEqual(len(values), len(set(values)))

    def test_compute_metrics_counts_missing_predictions_as_wrong(self):
        predictions = {
            "q1.jpg": ["g1.jpg", "g2.jpg"],
            "extra.jpg": ["g9.jpg"],
        }
        ground_truth = {
            "q1.jpg": ["g1.jpg"],
            "q2.jpg": ["g3.jpg"],
            "q_empty.jpg": [],
        }
        metrics = compute_metrics(predictions, ground_truth)
        self.assertEqual(metrics["evaluated"], 2)
        self.assertEqual(metrics["missing_predictions"], 1)
        self.assertEqual(metrics["extra_predictions"], 1)
        self.assertAlmostEqual(metrics["top1"], 0.5)
        self.assertAlmostEqual(metrics["top5"], 0.5)
        self.assertAlmostEqual(metrics["top10"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

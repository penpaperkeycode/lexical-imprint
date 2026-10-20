"""DSIR — Data Selection via Importance Resampling (Xie et al., NeurIPS 2023).

Top-K deterministic variant + multi-pod parallel implementation.

Reproducing paper §5.2:
    from dsir.select import select_topk_indices
    indices = select_topk_indices(source_corpus, target_corpus, K=67_869_569,
                                  bucket_count=10_000, n_max=2)
"""
from .hashing import hash_ngrams, ngram_iter, normalize_text
from .counting import count_buckets, count_buckets_streaming
from .scoring import compute_log_ratio, score_text, top_k_heap
from .select import select_topk_indices

__version__ = "0.1.0"

__all__ = [
    "hash_ngrams",
    "ngram_iter",
    "normalize_text",
    "count_buckets",
    "count_buckets_streaming",
    "compute_log_ratio",
    "score_text",
    "top_k_heap",
    "select_topk_indices",
]

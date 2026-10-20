"""Word-level Unigram Cross-Entropy (UCE) metric.

UCE measures alignment between a pretraining corpus's unigram distribution
and a benchmark's unigram distribution:

    UCE(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) log P_pre(w)

Lower UCE = the pretraining corpus better covers the benchmark vocabulary.
"""
from .compute_unigram_ce import (
    tokenize,
    normalize_text,
    build_benchmark_q,
    build_train_counts_parquet,
    compute_uce_nats,
    nats_to_bits,
)

__all__ = [
    "tokenize",
    "normalize_text",
    "build_benchmark_q",
    "build_train_counts_parquet",
    "compute_uce_nats",
    "nats_to_bits",
]

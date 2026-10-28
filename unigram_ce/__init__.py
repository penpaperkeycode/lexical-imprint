"""Word-level Unigram Cross-Entropy (UCE) metric.

UCE measures alignment between a pretraining corpus's unigram distribution
and a benchmark's unigram distribution:

    UCE(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) log P_pre(w)

Lower UCE = the pretraining corpus better covers the benchmark vocabulary.

`compute_uce_nats` returns nats; `nats_to_bits` converts to bits, the unit of
the paper. `build_train_counts_parquet` starts worker processes, so scripts
must call it under `if __name__ == "__main__":` (spawn and forkserver start
methods re-import the script in each worker).
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

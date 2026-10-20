# Word-level Unigram Cross-Entropy (UCE)

This module implements the unigram cross-entropy metric described in §3
of the paper:

    UCE(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) log P_pre(w)

Inputs:
- A benchmark text source (parquet, column `text`)
- A pretraining-corpus text source (parquet, column `text`)

Output:
- Single scalar UCE value in nats (and bits with `--bits`)

The reported value is interpretable as the average bits required to
encode each benchmark token under the pretraining-corpus unigram
distribution. Lower = the corpus better matches benchmark vocabulary.

## Tokenization

Identical pipeline as `ce_filter`: NFKC normalization, punctuation
stripping (non-word characters replaced by spaces), and whitespace
splitting. Words are not lowercased, as in the paper. Pass `--lowercase`
to lowercase them first (the paper's appendix reports this as a
preprocessing ablation). Corpus counts of benchmark words that never
occur in the corpus are floored at one, so every benchmark word has a
non-zero probability.

## Quick example

```bash
python compute_unigram_ce.py \
    --bench_path /path/to/benchmarks/arc_easy/test.parquet \
    --bench_col text \
    --train_path /path/to/pretraining_corpus/ \
    --train_col text \
    --n_workers 64 \
    --bits
```

Output:

```
UCE = 8.7421 nats   (12.6171 bits)
```

## Library usage

```python
from unigram_ce import (
    build_benchmark_q,
    build_train_counts_parquet,
    compute_uce_nats,
    nats_to_bits,
)

q, vocab = build_benchmark_q("benchmarks/arc_easy/test.parquet")
counts, n = build_train_counts_parquet("pretrain_corpus/", text_col="text",
                                        vocab=vocab, n_workers=64)
print(compute_uce_nats(q, counts, n))
```

## Reproducing paper §4 numbers

For each (corpus, benchmark) pair in the paper's corpus-level results
(Figure 2 and the per-task tables in the appendix), run the script once
and record the value. Total wall-clock time scales linearly with
the corpus size; on 64 CPU cores, scanning 100B-token DCLM takes ~3-4 hours.

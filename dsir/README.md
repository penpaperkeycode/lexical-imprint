# DSIR — top-K deterministic baseline (NeurIPS 2026 §5.2)

Implementation of **DSIR** (Data Selection for Language Models via Importance
Resampling — Xie et al., NeurIPS 2023) used as a baseline in our paper's §5.2.

We use the **top-K deterministic** variant (rather than DSIR's default Gumbel-top-K
sampling) to match our paper's CE-filter framework — both methods select a fixed
number of source rows by an analytical alignment score.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/                       # ~1 minute
bash examples/tiny_demo/run.sh      # 30-second end-to-end demo
```

## Reproducing paper §5.2

For each source corpus (DCLM, FineWeb-Edu) we run a 5-phase pipeline:

| Phase | What | Pods |
|---|---|---|
| A | target n-gram bucket counts | 1 |
| B | source n-gram bucket counts (4-way split) | **4 parallel** |
| C | merge counts → log P_t / P_s | 1 |
| D | score rows + per-pod top-K heap (4-way split) | **4 parallel** |
| E | merge 4 heaps → global top-K → write filtered parquets | 1 |

**Single machine** (small/medium corpora):

```bash
bash scripts/run_dsir_select.sh \
    --source <dir-of-parquets> \
    --target <benchmark-dir> \
    --out_dir <out-parquets> \
    --K <top-K> \
    --bucket_count 10000 --n_max 2
```

**K8s multi-pod** (4-pod parallel B/D):

```bash
SOURCE_TAG=fwedu YAML_DIR=/path/to/jobs/dsir \
    bash scripts/run_dsir_select_k8s.sh
```

The driver expects per-phase PyTorchJob/Job YAMLs at `${YAML_DIR}/`
following the naming convention shown in
`scripts/run_dsir_select_k8s.sh`. Cluster-specific YAMLs (image,
namespace, node selectors, storage mounts) are not bundled with this
release — substitute your own.

## Algorithm (precise)

For each row $x$ in the source corpus we score:

$$
\log\,iw(x) \;=\; \sum_{g \in \text{ngrams}(x)} \log P_t(\phi(g)) - \log P_s(\phi(g))
$$

where $\phi: \text{ngram} \to \{0, \dots, B-1\}$ is murmurhash3 mod $B$,
$P_t / P_s$ are Laplace-smoothed (add-1) bucket multinomials over the target /
source corpora, and ngrams = unigram + bigram (after NFKC + lowercase + whitespace
tokenization). We then keep the **top-K rows by $\log iw(x)$**.

For the paper:
- $B = 10\,000$ (DSIR paper default)
- $n_{\max} = 2$ (uni + bi)
- smoothing = 1.0
- DCLM: $K = 126\,014\,911$ (top 20% of 630M rows)
- FWEDU: $K = 67\,869\,569$ (top 20% of 339M rows)

Training hyperparameters are **identical** to the CE-filter and Random baselines:
70B GPT-2 tokens, GBS=256, MBS=16, seq=2048, BF16, AdamW($\beta=0.9, 0.95$),
WD=0.033, clip=1.0, peak LR $1e-3 \to 1e-5$ cosine, 5000-iter warmup.

## Layout

```
dsir/
├── dsir/                       # importable Python package
│   ├── hashing.py              # mmh3 + ngram extraction
│   ├── counting.py             # streaming bucket counts
│   ├── scoring.py              # log_iw + heap top-K
│   ├── select.py               # high-level select_topk_indices()
│   └── parquet_io.py           # large_string-safe pyarrow streaming
├── scripts/                    # 5-phase CLI for multi-pod
│   ├── target_count.py         # Phase A
│   ├── source_count_shard.py   # Phase B
│   ├── merge_log_ratio.py      # Phase C
│   ├── score_shard.py          # Phase D
│   ├── finalize.py             # Phase E
│   ├── finalize_fast.py        # Phase E fast path (per-worker partials)
│   ├── consolidate_partials.py # Phase E offline consolidation
│   ├── emit_filtered.py        # Phase E.2 (multi-pod variant)
│   ├── merge_global_topk.py    # Global top-K merge utility
│   ├── split_source_shards.py  # Round-robin shard splitter
│   ├── run_dsir_select.sh      # single-machine driver
│   └── run_dsir_select_k8s.sh  # K8s driver (apply + wait gates)
├── examples/tiny_demo/         # 30-sec self-contained demo
├── tests/                      # pytest unit tests
└── docs/                       # algorithm / orchestration / reproducibility
```

## Determinism

- `mmh3.hash(.., signed=False)` is byte-stable across Python 3.x (mmh3 ≥ 4.0).
- Top-K is deterministic — no random sampling anywhere.
- Source/target shard splitting is lexicographic (sorted file paths, round-robin).
- `examples/tiny_demo/expected_output.json` records SHA256 of input parquets +
  exact selected indices for verification.

## Reference

Xie, S. M., Santurkar, S., Ma, T., & Liang, P.\ (2023).
*Data Selection for Language Models via Importance Resampling.*
Advances in Neural Information Processing Systems 36 (NeurIPS 2023).

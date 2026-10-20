# Reproducibility — paper §5.2

## Source corpora versions

- **DCLM**: `mlfoundations/dclm-baseline-1.0-parquet`, shards `00–03` (4 of 100
  global shards). 630 074 555 rows post-load.
- **FineWeb-Edu**: `HuggingFaceFW/fineweb-edu`, the **350BT** subset.
  339 347 847 rows post-load.

Replace with HF revision SHAs in the final paper:

```python
load_dataset("mlfoundations/dclm-baseline-1.0-parquet", revision="<sha>")
load_dataset("HuggingFaceFW/fineweb-edu", "sample-350BT", revision="<sha>")
```

## Target corpus

CSR7 benchmark group at `ce_filter/benchmark_group_csr7/`:
- arc_easy
- arc_challenge
- hellaswag
- sciq
- openbookqa
- piqa
- swag

Each task's HuggingFace dataset is concatenated into a single parquet of `text`
column.

## DSIR hyperparameters (frozen)

| name | value |
|---|---|
| bucket_count B | 10 000 |
| n_max | 2 (uni + bi) |
| smoothing | 1.0 (Laplace add-1) |
| K (FWEDU) | 67 869 569 (top 20%) |
| K (DCLM) | 126 014 911 (top 20%) |
| Hash | mmh3.hash(.., signed=False) % B |
| Tokenization | whitespace split after NFKC + lowercase |

## Random seeds

- DSIR top-K is deterministic — no random sampling.
- Source/target shard splitting is round-robin over the lexicographic file list.
- Training runs use seed 1234 (matching the CE-filter and Random baselines).
- Random baseline uses seed 42 for its 20% Bernoulli subsample.

## Environment

- Python 3.11 inside the training container.
- `mmh3==4.1.0`, `pyarrow==15.0.2`, `numpy==1.26.4`.
- For training: NVIDIA Megatron-LM v0.15.0, BF16, flash-attn 2.x.

## Verification

`examples/tiny_demo/expected_output.json` records:
- SHA256 checksums of the synthetic source/target parquets.
- Exact list of selected indices for $K=100$.

A re-run that matches both is byte-identical to the reference.

## Paper artifacts

- 4 trained DSIR models: 1.34B {DCLM, FWEDU} + 3.33B {DCLM, FWEDU}.
- Each at iter 133 514 (= 70B GPT-2 tokens / GBS 256 / seq 2048).
- Evaluated at zero-shot on 11 lm-evaluation-harness tasks (CSR7 + 4 OOD).

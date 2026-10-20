# Benchmark preprocessing — CSR7 + OOD reference

This module downloads the 11 evaluation benchmarks used in the paper
(§5.2) from the HuggingFace Hub and writes one parquet per benchmark.
The output is the input to:

- `unigram_ce/compute_unigram_ce.py` (UCE metric on benchmark text)
- `ce_filter/ce_filter.py` (CE-filter benchmark scan; uses the combined
  `benchmark_group_en/` directory by default)

## Tasks (11 total)

**CSR7 (CE-filter targets):** ARC-E, ARC-C, HellaSwag, SciQ, OpenBookQA,
PIQA, SWAG.

**OOD reference (not targeted by CE filter, used to validate generalization):**
LAMBADA, CommonsenseQA, BoolQ, Social-IQA.

## Quick start

Download all 11 benchmarks:

```bash
python preprocess_csr7.py --output_dir /data/benchmarks/
```

Outputs:

```
/data/benchmarks/
  arc_easy/arc_easy.parquet
  arc_challenge/arc_challenge.parquet
  hellaswag/hellaswag.parquet
  ... (8 more)
  benchmark_group_en/
    arc_easy.parquet
    arc_challenge.parquet
    ... (combined directory; CE filter input)
```

The combined `benchmark_group_en/` directory is the standard input to
`ce_filter.py`'s `--bench_path`.

Subset of tasks:

```bash
python preprocess_csr7.py --output_dir /data/benchmarks/ \
    --tasks arc_easy,arc_challenge,piqa
```

## Text format

For each example, the output `text` field concatenates the question
(or context) with all answer-choice strings, separated by spaces. After
concatenation, punctuation is bracketed by newlines (matching the paper's
preprocessing). The file is then a flat one-column parquet of strings,
one example per row.

This format is suitable for unigram-frequency analysis: every word the
model would see when answering the question is included exactly once in
the text representation, so the unigram distribution faithfully reflects
the linguistic content of the benchmark.

## Reproducibility

All datasets are downloaded from public HuggingFace mirrors. No Hub
authentication is required for the listed datasets. If a dataset
requires `trust_remote_code=True` (PIQA), the script passes it.

## Dependencies

- `datasets >= 2.14`
- `pyarrow >= 14`
- `huggingface_hub` (transitively via `datasets`)

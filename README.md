<h1 align="center">The Lexical Imprint of Pre-training Data</h1>

<p align="center">
  <a href="https://openreview.net/forum?id=8hN2Bcw0x2"><img src="https://img.shields.io/badge/OpenReview-8hN2Bcw0x2-8c1b13.svg" alt="OpenReview"></a>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/155069"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg" alt="NeurIPS 2026"></a>
  <a href="https://arxiv.org/abs/2602.10657"><img src="https://img.shields.io/badge/arXiv-2602.10657-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  Jeonghoon Kim<sup>1,2*</sup> · Woojin Chung<sup>1*†</sup> · Woomin Song<sup>2</sup> · Cheonbok Park<sup>1,2</sup> · Nako Sung<sup>1</sup> · Jinwoo Shin<sup>2</sup><br>
  <sup>1</sup>NAVER Cloud &nbsp;&nbsp; <sup>2</sup>KAIST &nbsp;&nbsp; <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Work done during an internship at NAVER Cloud
</p>

<p align="center">
  <img src="assets/teaser.png" width="92%" alt="Unigram cross-entropy at the corpus, document, and prompt scopes">
</p>

This repository contains the code for the NeurIPS 2026 paper **The Lexical Imprint of Pre-training Data**.

Word-level unigram cross-entropy (UCE) measures how closely the word distribution of a reference corpus matches that of a target benchmark. In controlled from-scratch pre-training on four corpora at three model scales, lower UCE goes with higher zero-shot accuracy on each of the 10 benchmarks in this sweep. Selecting training documents that lower UCE against a target (the UCE-filter) raises accuracy on that target above a random selection at the same token budget. Routing each prompt to the drafter whose training corpus has the lowest UCE against it raises speculative-decoding throughput.

The repository provides the paper's components: the UCE metric, the UCE-filter, and the benchmark preprocessing that builds the target word distribution *Q*. Each comes with tests.

## Contents

| Path | What it does | Paper |
|---|---|---|
| [`unigram_ce/`](unigram_ce) | Word-level unigram cross-entropy between a pre-training corpus and one benchmark or a pooled group of benchmarks | §3, §4, App. A.1 |
| [`ce_filter/`](ce_filter) | UCE-filter (named CE-filter in the code): analytic row-local ΔH scoring that removes the `--top_pct`% least-aligned rows (paper: 80, keeps 20%) | §5.1, §5.2 |
| [`benchmarks/`](benchmarks) | Downloads the benchmark text that defines *Q*: the CSR7 targets and the four out-of-target tasks (no MMLU) | §4.1, §5.2 |
| [`tests/`](tests) | pytest tests for the three components (no network needed) | |

`unigram_ce/`, `ce_filter/` and `benchmarks/` each have a `README.md` that explains their options. `--help` on each script lists them all.

## Installation

The code needs Python 3.9 or later.

```bash
git clone https://github.com/penpaperkeycode/lexical-imprint.git
cd lexical-imprint
pip install -r requirements.txt
```

## Quick start

Run the commands from the repository root. `/data/cc_dump/` stands for a pre-training corpus: a parquet file or a directory of parquet files with a `text` column.

Build the CSR7 target group from the Hugging Face Hub (no login needed):

```bash
python benchmarks/preprocess_csr7.py \
    --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag \
    --output_dir /data/csr7/
```

Measure UCE between the corpus and one benchmark, in bits:

```bash
python unigram_ce/compute_unigram_ce.py \
    --bench_path /data/csr7/arc_easy/arc_easy.parquet \
    --train_path /data/cc_dump/ \
    --n_workers 64 --bits
```

Run the UCE-filter against the pooled CSR7 target. `--top_pct 80` removes the 80% of rows least aligned with *Q* and keeps 20%, the paper setting:

```bash
python ce_filter/ce_filter.py \
    --bench_path /data/csr7/benchmark_group_en/ \
    --train_path /data/cc_dump/ \
    --train_format parquet \
    --out_parquet /data/ucef_keep20.parquet \
    --n_workers 120 \
    --skip_seq_apply \
    --top_pct 80
```

The first command writes `/data/csr7/<task>/<task>.parquet` for each task and the combined directory `/data/csr7/benchmark_group_en/` with the same seven files. A directory passed as `--bench_path` is pooled into one *Q*, weighted by word count, so the UCE-filter targets the seven tasks together. Pass the combined directory, not the output root.

The UCE script prints `UCE = <x> nats   (<y> bits)`. The UCE-filter writes the kept rows, in corpus order and with all their columns, to one parquet file (`--out_num_shards N` writes up to N part files into a directory instead). It prints `H_before`, `H_after` and the share of rows removed. `--skip_seq_apply` skips only a single-threaded consistency check that re-reads the corpus. The written rows are the same with or without it. On large corpora the check takes longer than the selection itself, and the paper's runs used the flag. The worker counts above match our machines. Set `--n_workers` to your number of CPU cores (`ce_filter.py` caps it at that number).

To re-measure UCE on the written subset:

```bash
python ce_filter/ce_filter.py \
    --bench_path /data/csr7/benchmark_group_en/ \
    --out_parquet /data/ucef_keep20.parquet \
    --verify_only
```

This prints `H_after (recomputed)` in bits and does not need `--train_path`. Running `unigram_ce/compute_unigram_ce.py` with the same `--bench_path`, `--train_path /data/ucef_keep20.parquet` and `--bits` gives the same value. For the out-of-target UCE of the subset, build the four out-of-target tasks into a second directory (`--tasks lambada_openai,commonsense_qa,boolq,social_iqa --output_dir /data/oot4/`) and pass `/data/oot4/benchmark_group_en/` as `--bench_path`.

## Method summary

**UCE.** For a target benchmark with word distribution *Q* and a reference corpus with word distribution *P*,

    UCE(Q, P) = - sum_{w in V_b} Q(w) log P(w) = H(Q) + KL(Q || P),

where *V_b* is the benchmark vocabulary. Corpus counts of benchmark words that never occur in the corpus are floored at one: P(w) = max(c(w), 1) / (T + B), where T counts all word tokens of the corpus and B is the number of benchmark words with c(w) = 0 (Appendix A.1). Positive counts are unchanged, so this is not Laplace smoothing. Words come from NFKC normalization, punctuation stripping, and whitespace splitting, without lowercasing, as in the paper. The UCE script and the UCE-filter share this tokenizer, and `--lowercase` in `unigram_ce/compute_unigram_ce.py` lowercases words first (a preprocessing ablation in the paper's appendix). The paper reports bits. `compute_unigram_ce.py` prints nats and adds bits with `--bits`. `ce_filter.py` prints bits.

**UCE-filter.** For each training row *r*, the change in UCE from removing *r* alone, ΔH(*r*), has a closed form that depends only on the words in *r* and the corpus counts. One pass counts the benchmark words in the corpus, and a second pass scores every row. The `--top_pct` percent of rows with the smallest ΔH, the rows whose removal lowers UCE the most, are then removed together. The remaining rows are the share most aligned with *Q* and are written in their original order. The paper removes 80% of the rows and keeps 20% (`--top_pct 80`, the default). Rows with null text are never scored and always kept. Ties at the cutoff are broken by row index, so the output is deterministic. In the paper's runs, with `--skip_seq_apply` on one 120-core machine, a selection took about 3.5 hours on the FineWeb-Edu pool and 8.3 hours on the DCLM pool. The removal set stays in memory. At `--top_pct 80` it takes about 44 GB of RAM for a 339M-row corpus (the paper's FineWeb-Edu pool) and about 82 GB for 630M rows (DCLM).

**Benchmarks.** `benchmarks/preprocess_csr7.py` downloads the benchmarks of the UCE-filter study from the Hugging Face Hub. It writes one parquet file per task and a combined `benchmark_group_en/` directory that holds the tasks written by that run. The targeted group (CSR7) is ARC-Easy, ARC-Challenge, HellaSwag, SciQ, OpenBookQA, PIQA, and SWAG. The out-of-target group (OOT-4) is LAMBADA, CommonsenseQA, BoolQ, and Social IQA. Each row's text joins the raw dataset fields. For most tasks this is the question or context followed by all answer choices. SciQ adds the support passage, BoolQ is the passage and the question, and LAMBADA is the passage. HellaSwag and CommonsenseQA are read from their train and validation splits. The paper's *Q* also pooled their test splits, which moves UCE by 0.01 to 0.02 bits on our corpus samples. MMLU, which Section 4 also uses, is not included.

## Scope

The repository provides the paper's components, not a pipeline that reproduces every number in the paper.

Included:
- the UCE metric (`unigram_ce/`), the UCE-filter (`ce_filter/`), and the benchmark preprocessing that builds *Q* (`benchmarks/`)
- tests for all three (`tests/`)

Not included:
- the baseline selections (DSIR, BETR) and the random control
- pre-training and evaluation (the paper's appendix gives the training and evaluation configuration)
- the inference-time routing study (Section 6)
- the 30B case study (Section 5.3)
- scripts for the appendix analyses (most of them combine the functions here)

## Testing

```bash
python -m pytest
```

`pytest.ini` runs the tests in `tests/` with the repository root on the import path. The tests build toy data in temporary directories and need no network. The benchmark tests replace the Hugging Face download with a stub. The full run takes one to three minutes on a laptop, most of it in the UCE-filter tests, which start worker processes. To test one component, pass its file, for example `python -m pytest tests/test_ce_filter.py`.

## Changes from the submitted package

The code differs from the supplementary package submitted to OpenReview as follows.

- The DSIR baseline (`dsir/`) is removed, so the repository holds only the paper's own components.
- `ce_filter.py` no longer shifts row numbers when the corpus has rows with null text, so the rows it removes are the rows it selected.
- `ce_filter.py` and `compute_unigram_ce.py` now stop with exit code 1 and a one-line error message when an input file is missing or unreadable, lacks the text column, or is a broken symlink, and when the corpus has no words. `ce_filter.py` also stops when the rows it reads, scores and writes do not add up.
- `preprocess_csr7.py` exits with code 1 when a task fails, for example on a download error or a missing split, and still writes the other tasks.
- `ce_filter.py --resume` refuses to continue from checkpoints written with other settings or from changed input files.
- `--top_pct` in `ce_filter.py` has help text and a range check, and its default is the paper setting 80 (it was 0.5).
- `preprocess_csr7.py` reads PIQA from the Hub's parquet conversion, so PIQA also loads with `datasets` 4.0 and later, where the submitted script skipped it.
- Rows with equal ΔH at the cutoff are removed in a fixed order (larger row index first), so repeated runs write the same rows.
- `ce_filter.py --verify_only` no longer needs `--train_path`.
- `ce_filter.py` starts at most one worker process per CPU, so the default `--n_workers 120` also runs on smaller machines.
- `compute_unigram_ce.py` finds input files with the same rule as `ce_filter.py` (recursive, skipping names that start with `.` or `_`), so both tools read the same files.
- Each `preprocess_csr7.py` run removes task files of earlier runs from `benchmark_group_en/` and warns about any other file there that `--bench_path` would read, because those files are pooled into *Q* too.
- The READMEs and docstrings now describe `--top_pct` as the percent of rows removed, the smoothing as a zero-count floor rather than Laplace smoothing, and the UCE-filter target as the CSR7 group, and their example commands run as written.
- `compute_unigram_ce.py` lowercased words in the submitted version, while the paper's numbers use case-preserving words, so it now preserves case by default and keeps the old behavior behind `--lowercase`.
- Author names replace the anonymous placeholders in the license file.

## Citation

```bibtex
@inproceedings{kim2026lexical,
  title     = {The Lexical Imprint of Pre-training Data},
  author    = {Kim, Jeonghoon and Chung, Woojin and Song, Woomin and Park, Cheonbok and Sung, Nako and Shin, Jinwoo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

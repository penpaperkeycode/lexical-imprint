# Word-level Unigram Cross-Entropy (UCE)

This module implements the unigram cross-entropy metric described in §3
of the paper:

    UCE(P_eval, P_pre) = - sum_{w in V_eval} P_eval(w) log P_pre(w)

Inputs:
- A benchmark text source: one parquet file, or a directory of parquet
  files pooled into one distribution (column `text`)
- A pretraining-corpus text source: one parquet file or a directory of
  parquet files (column `text`)

Output:
- One UCE value, printed in nats. Pass `--bits` to also print it in
  bits, the unit used in the paper. Without `--bits` only nats are
  printed.

UCE is the average number of nats (bits with `--bits`) needed to encode
a benchmark word under the pretraining-corpus unigram distribution.
Lower = the corpus better matches benchmark vocabulary.

## Tokenization

Identical pipeline as `ce_filter`: NFKC normalization, punctuation
stripping (non-word characters replaced by spaces), and whitespace
splitting. Words are not lowercased, as in the paper. Pass `--lowercase`
to lowercase them first (the paper's appendix reports this as a
preprocessing ablation).

## Smoothing

Corpus counts of benchmark words that never occur in the corpus are
floored at one, so every benchmark word has a non-zero probability
(appendix A.1):

    P_pre(w) = max(c(w), 1) / (T + B)

- `c(w)`: count of `w` in the corpus
- `T`: total number of word tokens in the corpus (all words, not only
  benchmark words)
- `B`: number of benchmark words with `c(w) = 0`

Positive counts are unchanged, so this is not Laplace (add-one)
smoothing.

## Quick example

From the repository root. Build the benchmark files, here the seven
CSR7 tasks (`benchmarks/README.md` lists all tasks):

```bash
python benchmarks/preprocess_csr7.py \
    --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag \
    --output_dir /data/csr7/
```

This writes one file per task, `/data/csr7/<task>/<task>.parquet`, and
the pooled group `/data/csr7/benchmark_group_en/` (see
[Grouped targets](#grouped-targets)). Measure UCE between one task and
a pretraining corpus at `/data/cc_dump/`, a parquet file or a directory
of parquet files:

```bash
python unigram_ce/compute_unigram_ce.py \
    --bench_path /data/csr7/arc_easy/arc_easy.parquet \
    --bench_col text \
    --train_path /data/cc_dump/ \
    --train_col text \
    --n_workers 64 \
    --bits
```

Output:

```
UCE = 8.7421 nats   (12.6122 bits)
```

The result line goes to stdout; progress lines, including the number of
benchmark and corpus files read, go to stderr.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--bench_path` | required | Benchmark parquet file, or a directory whose files are pooled into one Q |
| `--bench_col` | `text` | Text column of the benchmark files |
| `--train_path` | required | Pretraining-corpus parquet file or directory |
| `--train_col` | `text` | Text column of the corpus files |
| `--n_workers` | the CPU count | Worker processes for the corpus scan, at most one per corpus file. Affects only speed, not the result |
| `--bits` | off | Also print UCE in bits, the unit of the paper |
| `--lowercase` | off | Lowercase words before counting (off in the paper) |

## Input files

`--bench_path` and `--train_path` take a parquet file or a directory. A
directory is searched recursively with pyarrow dataset discovery, the
same rule `ce_filter` uses, so both tools read the same files: names
starting with `.` or `_` are skipped (hidden files, Spark `_SUCCESS` and
`_temporary/`), and every other file must be parquet. For a Hugging Face
dataset download, pass its data directory (for example
`<download>/data/`), not the repository root, whose `README.md` is not
parquet. Symlinks are followed, so a directory of symlinks to parquet
files works.

The run stops with a non-zero exit and a one-line `[UCE] error: ...`
message naming the path when a path does not exist, a file cannot be
read, a file has no `--bench_col` / `--train_col` column, a symlink is
broken, the same file is reached through more than one path (for
example through a symlink loop), or the corpus has no word tokens.

## Grouped targets

A directory `--bench_path` pools the rows of all its parquet files into
one Q, so each task is weighted by its number of word tokens. The
paper's grouped values, such as the 7-task CSR7 target of §5, are
computed this way. Do not average per-task UCE values instead: on a
33.5M-word C4 sample, the plain mean of the seven CSR7 per-task values
is 0.28 bits higher than the pooled value (11.50 vs 11.22 bits).

To measure UCE against the CSR7 group built in the quick example:

```bash
python unigram_ce/compute_unigram_ce.py \
    --bench_path /data/csr7/benchmark_group_en/ \
    --train_path /data/cc_dump/ \
    --n_workers 64 \
    --bits
```

The run should report 7 benchmark parquet files. The out-of-target group
is built the same way, with
`--tasks lambada_openai,commonsense_qa,boolq,social_iqa --output_dir /data/oot4/`,
and passed as `--bench_path /data/oot4/benchmark_group_en/`.

## Library usage

```python
from unigram_ce import (
    build_benchmark_q,
    build_train_counts_parquet,
    compute_uce_nats,
    nats_to_bits,
)

if __name__ == "__main__":
    q, vocab = build_benchmark_q("/data/csr7/arc_easy/arc_easy.parquet")
    counts, n_tokens = build_train_counts_parquet("/data/cc_dump/",
                                                  text_col="text",
                                                  vocab=vocab, n_workers=64)
    h = compute_uce_nats(q, counts, n_tokens)
    print(f"UCE = {h:.4f} nats ({nats_to_bits(h):.4f} bits)")
```

The `if __name__ == "__main__":` guard is required because
`build_train_counts_parquet` starts worker processes, and under the
spawn and forkserver start methods (macOS, Windows, and Linux from
Python 3.14) each worker re-imports the script. Save the script in the
repository root, or add the root to `PYTHONPATH`, so that `unigram_ce`
can be imported. `compute_uce_nats` returns nats; `nats_to_bits`
converts to bits.

## Paper setting

§4 reports one value per (pretraining corpus, benchmark) pair: run the
script once per pair with the task's single file as `--bench_path`. The
UCE-filter results in §5 use grouped targets (see above). All values in
the paper are in bits (`--bits`). The pretraining corpora are not part of
this repository, and the benchmark files built here differ slightly from
the paper's for some tasks (see `benchmarks/README.md`), so values
computed here can differ from the printed ones. Wall-clock time scales
linearly with the corpus size.

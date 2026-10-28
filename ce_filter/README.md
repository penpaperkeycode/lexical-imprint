# UCE-filter — data selection by ΔH (paper §5.1)

The UCE-filter is called CE-filter in the code. `ce_filter.py` runs a
6-step pipeline:

1. Scans benchmark text and builds the target unigram distribution Q.
2. Scans the pre-training corpus and counts benchmark words and the
   total number of word tokens.
3. Sets K, the number of rows to remove, from `--top_pct`.
4. Scores every row in closed form (analytic ΔH, O(unique words in the
   row) per row) and selects the K rows with the smallest ΔH, the rows
   least aligned with Q.
5. Writes all other rows, in their original order, to one parquet file
   or to N shards.
6. Recomputes UCE on the written output and prints it next to the
   unfiltered value (sanity check).

The selection is one-shot: every row is scored against the full corpus
counts, and the K rows are removed together.

## Quick example

From the repository root. First build Q from the 7 CSR7 tasks, the
paper's UCE-filter target, in a fresh output directory:

```bash
python benchmarks/preprocess_csr7.py \
    --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag \
    --output_dir /data/csr7/
```

Its `benchmark_group_en/` directory then holds the 7 task files.
Filter the corpus `/data/cc_dump/` against it. `--top_pct 80` removes
the 80% of rows least aligned with Q and keeps 20%, the paper setting.
This example writes the kept rows as 12 shard files into
`/data/ucef_keep20_shards/`:

```bash
python ce_filter/ce_filter.py \
    --bench_path /data/csr7/benchmark_group_en/ \
    --bench_col  text \
    --train_path /data/cc_dump/ \
    --train_col  text \
    --train_format parquet \
    --out_parquet  /data/ucef_keep20_shards/ \
    --out_num_shards 12 \
    --n_workers   120 \
    --skip_seq_apply \
    --resume \
    --top_pct 80
```

Without `--out_num_shards` the output is one file, for example
`--out_parquet /data/ucef_keep20.parquet`.

A directory passed as `--bench_path` is pooled into one token-weighted
Q: the rows of all its files are counted together, so a task with more
words weighs more. Q is not an average of per-task distributions.

`--train_format` may be `parquet` or `wet` (raw Common Crawl WET files,
decoded with `warcio`); it is detected from the file names if omitted.
`--bench_format` does the same for `--bench_path`.

`--n_workers` (default 120) is the number of worker processes. It is
capped at the number of CPUs, so the default also runs on smaller
machines; set it to your number of CPU cores. Each scan also uses at
most one worker per file, and the Step 6 scan at most 20. A value
below 1 is an error. `--worker_batch` (default 1024) and
`--batch_size` (default 8192) only affect speed, not the result.
`--progress_every` (default 50000) sets how often, in rows, a progress
line is printed to stderr.

## `--top_pct`

`--top_pct` is the percent of rows REMOVED. The removed rows are those
with the smallest ΔH, so `--top_pct 80` removes 80% of the rows and
keeps the 20% most aligned with Q. This is the paper setting and the
default.

- K = `max(1, int(N_text * top_pct / 100))`, at most N_text, where
  N_text is the number of rows with non-null text. `--top_pct 0`
  removes nothing.
- Allowed range: 0 <= top_pct < 100; other values stop with an error
  (not checked with `--verify_only`, which does not use it).
- A value between 0 and 1 prints a warning: `--top_pct 0.8` removes
  0.8% of rows, not 80%.

The summary prints the share actually removed, as
`rows removed = <removed> / <all rows> (<share>%)`.

## Null text and ties

- Rows whose text is null are never scored and never removed; they are
  written unchanged. K counts only rows with text, so null rows do not
  change which text rows are selected, and the printed share is then a
  little below `--top_pct`. Empty strings are ordinary rows.
- Rows are removed from the smallest ΔH up. Among rows with equal ΔH
  at the cutoff, the row with the larger row index (later in the
  corpus) is removed first, so repeated runs write the same rows
  whatever the worker scheduling.

## Errors

The run stops with exit code 1 and a one-line `[error] ...` message
in these cases:

- An input path does not exist, a benchmark or corpus file cannot be
  opened as parquet (for example a corrupt footer or a file that is
  not parquet), a file in an input directory is a broken symlink, or a
  symlinked directory makes the same file appear more than once (a
  symlink loop). The message names the file or directory. This also
  covers `--verify_only` with a missing `--out_parquet`.
- A benchmark or corpus file has no `--bench_col` / `--train_col`
  column, or a scan (Steps 1, 2 and 6) cannot read it, for example a
  corrupt text column or a truncated WET file. Every parquet file needs
  the column, empty files included. The message names the file.
- The benchmark or the corpus has no word tokens. The message names
  the column and the path.
- The row counts do not add up: the scoring pass did not see every
  corpus row (scored rows plus null rows must equal the row count), the
  writer did not write exactly the rows that were not removed, or the
  seq-apply check below failed. The first two checks also run with
  `--skip_seq_apply`.

Steps 1-4 read only the text column. A corrupt column other than the
text column is first read in Step 5, which copies every column. That
also stops the run with one line and exit code 1, but the message may
not name the file, and the partial output stays on disk.

## Output files

With `--out_num_shards 1` (default) the output is a single parquet file
at `--out_parquet`, such as `/data/ucef_keep20.parquet`. A trailing `/`
does not make it a directory: `/data/out/` becomes a file named `out`.
With `--out_num_shards N` (N > 1), `--out_parquet` is a directory
holding at most N files `part-0000.parquet`, `part-0001.parquet`, ...;
the kept rows are split in order, about kept/N rows per file. The
output keeps every column of the corpus (WET input gives the columns
`text` and `url`).

## Step 6 and `--skip_seq_apply`

Step 6 scans the written output and prints `H_after`. It then runs the
seq-apply check: it re-reads the corpus in a single thread, subtracts
the counts of the removed rows from the Step 2 counts, recomputes H and
stops with an error if it differs from `H_after` by more than 1e-9
bits. The check never changes the output; the written rows are the
same with or without it. On corpora of hundreds of millions of rows it
takes longer than the rest of the run, and the paper's selection runs
used `--skip_seq_apply`, which skips only this check. `--skip_step6`
skips all of Step 6.

## Runtime and memory

In the paper's runs, with `--skip_seq_apply` on one 120-core machine, a
selection took about 3.5 hours on the FineWeb-Edu pool and 8.3 hours
on the DCLM pool.

Step 4 keeps the K removed rows in a Python heap of (score, row index)
tuples and then builds a set of their row indices; both stay in memory
until the run ends. Measured costs are about 128.6 bytes per heap entry
plus 33.6 bytes per set entry. At `--top_pct 80` that is about 44 GB of
RAM for a corpus of 339M rows (the size of the paper's FineWeb-Edu
pool) and about 82 GB for 630M rows (DCLM). The need scales with K, so
a smaller `--top_pct` needs less.

## Resume / checkpointing

After Steps 3, 4 and 5 the script writes a checkpoint to
`{out_parquet}.ckpt/`, together with `run_config.json`, which records
`--bench_path`, `--bench_col`, `--train_path`, `--train_col`, both
formats, `--top_pct`, and a fingerprint of the benchmark and corpus
files (sorted file list with sizes and modification times, and the row
count). With `--resume`, a restarted run continues after the newest
checkpoint. It refuses to resume, with exit code 1 and a message naming
what changed, if any of these differ from the checkpointed run; delete
the `.ckpt` directory or run without `--resume`. A run without
`--resume` starts from Step 1 and replaces older checkpoints when it
reaches Step 3. The directory is removed after a successful run. The
Step 4 and 5 checkpoints hold the removal set, a few GB each at the
paper's scale.

## Re-measuring a written subset (`--verify_only`)

```bash
python ce_filter/ce_filter.py \
    --bench_path  /data/csr7/benchmark_group_en/ \
    --out_parquet /data/ucef_keep20_shards/ \
    --verify_only
```

`--verify_only` rebuilds Q from `--bench_path`, scans the parquet
output at `--out_parquet` (a shard directory as here, or a single file
such as `/data/ucef_keep20.parquet`) and prints its UCE as
`H_after (recomputed)`. It skips Steps 2-5 and does not need
`--train_path`. With the CSR7 group this is the targeted UCE of the
subset. To measure the same subset against the four out-of-target
tasks, build them into a second fresh directory and pass
`/data/oot4/benchmark_group_en/` as `--bench_path`:

```bash
python benchmarks/preprocess_csr7.py \
    --tasks lambada_openai,commonsense_qa,boolq,social_iqa \
    --output_dir /data/oot4/
```

## How the score is computed

Words come from NFKC normalization, replacing non-word characters with
spaces, and whitespace splitting, without lowercasing (the same
tokenizer as `unigram_ce`). Corpus counts of benchmark words that never
occur in the corpus are floored at one (appendix A.1):

    P(w) = max(c(w), 1) / (T + B)

Positive counts are unchanged, so this is not Laplace (add-one)
smoothing. For each row `r`, the score is the analytic change in UCE,
in bits, if `r` alone were removed:

    ΔH(r) = log(D_new / D)
          + sum_{w in row r} Q(w) · log( max(c(w),1) / max(c(w)−c_r(w),1) )

where:
- `c(w)` = count of benchmark word `w` in the corpus
- `c_r(w)` = count of `w` within row `r`
- `T` = number of word tokens in the corpus (all words, not only
  benchmark words); `B` = number of benchmark words with `c(w) = 0`
- `D = T + B`, `D_new = T − |r| + B_new`, where `|r|` counts all word
  tokens of `r` and `B_new` adds the benchmark words whose count drops
  to zero without `r`

Words not in row `r` cancel exactly, so the sum is only over the row's
unique words — no full-vocab scan, no Counter copies. The UCE change of
removing all K rows together (`ΔH (actual)` in the summary) differs
from the sum of their single-row scores because rows interact.

The derivation is in the docstring of `compute_delta_h_bits_analytical`
in `ce_filter.py`.

## Dependencies

```
pyarrow >= 14
warcio    (only for WET input)
```

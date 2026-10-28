# Benchmark preprocessing (target distribution Q)

`preprocess_csr7.py` downloads 11 benchmarks from the Hugging Face Hub and
writes one parquet file per benchmark with a single `text` column. These
files define the benchmark word distribution Q for:

- `unigram_ce/compute_unigram_ce.py` (UCE of a corpus against one benchmark
  or a group of benchmarks)
- `ce_filter/ce_filter.py` (UCE-filter; `--bench_path` is required and has
  no default)

## Tasks

**CSR7 (the UCE-filter target in the paper):** ARC-Easy, ARC-Challenge,
HellaSwag, SciQ, OpenBookQA, PIQA, SWAG.

**Out-of-target (not targeted by the UCE-filter):** LAMBADA, CommonsenseQA,
BoolQ, Social IQA.

| `--tasks` name | Hub dataset (config) | Splits read | Rows | Fields in `text`, in order |
|---|---|---|---:|---|
| `arc_easy` | `allenai/ai2_arc` (ARC-Easy) | train, validation, test | 5,197 | `question`, all `choices.text` |
| `arc_challenge` | `allenai/ai2_arc` (ARC-Challenge) | train, validation, test | 2,590 | `question`, all `choices.text` |
| `hellaswag` | `Rowan/hellaswag` | train, validation | 49,947 | `ctx`, all `endings` |
| `sciq` | `allenai/sciq` | train, validation, test | 13,679 | `question`, `correct_answer`, `distractor1`-`3`, `support` |
| `openbookqa` | `allenai/openbookqa` (main) | train, validation, test | 5,957 | `question_stem`, all `choices.text` |
| `piqa` | `ybisk/piqa` (revision `refs/convert/parquet`) | train, validation, test | 21,035 | `goal`, `sol1`, `sol2` |
| `swag` | `allenai/swag` (regular) | train, validation, test | 113,557 | `startphrase`, `ending0`-`3` |
| `lambada_openai` | `EleutherAI/lambada_openai` (en) | test | 5,153 | `text` |
| `commonsense_qa` | `tau/commonsense_qa` | train, validation | 10,962 | `question`, all `choices.text` |
| `boolq` | `google/boolq` | train, validation | 12,697 | `passage`, `question` |
| `social_iqa` | `lighteval/siqa` | train, validation | 35,364 | `context`, `question`, `answerA`-`C` |

Rows are the counts written on 2026-10-28. The datasets are loaded at their
current Hub version, so the counts can change if a dataset is edited on the
Hub. If a split listed above is missing from a dataset, that task fails and is
not written. PIQA is read from the Hub's parquet conversion because the main
branch of `ybisk/piqa` holds only a loading script, which `datasets` 4.0 and
later no longer run.

HellaSwag and CommonsenseQA are read from train and validation only. The
paper's Q also pooled their test splits (59,950 and 12,102 rows in total).
This changes Q only slightly: for the pooled CSR7 Q the total variation
distance is 0.018, and the UCE of FineWeb-Edu and C4 samples changes by 0.01
to 0.02 bits.

## Quick start

All 11 tasks:

```bash
python benchmarks/preprocess_csr7.py --output_dir /data/benchmarks/
```

Outputs:

```
/data/benchmarks/
  arc_easy/arc_easy.parquet
  arc_challenge/arc_challenge.parquet
  ... (one folder per task)
  benchmark_group_en/
    arc_easy.parquet
    arc_challenge.parquet
    ... (combined directory: the tasks written by this run)
```

For the UCE of one benchmark, pass its own file as `--bench_path`, e.g.
`/data/benchmarks/arc_easy/arc_easy.parquet`.

## CSR7 target group

The UCE-filter target in the paper is the pooled Q of the 7 CSR7 tasks. Build
it with `--tasks`:

```bash
python benchmarks/preprocess_csr7.py \
    --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag \
    --output_dir /data/csr7/
```

`/data/csr7/benchmark_group_en/` then holds exactly these 7 files.
Pass it as `--bench_path` to `ce_filter/ce_filter.py` (see
`ce_filter/README.md` for the other options) or to
`unigram_ce/compute_unigram_ce.py`. A directory passed as `--bench_path` is
read as one Q: the words of every parquet file under it are pooled, so each
task counts in proportion to its word count (HellaSwag and SWAG make up 83% of
the CSR7 words). The 11-task `benchmark_group_en/` from the quick start would
also target the 4 out-of-target tasks. Pass the combined directory, not the
output root, which also holds the per-task folders, including any left by
earlier runs.

Each run replaces the task files in `benchmark_group_en/` with the tasks
written in that run: task files left there by earlier runs are deleted (the
run log lists them). Files with other names are left alone, but `--bench_path`
pools them into Q too (it skips only names starting with `.` or `_`), so the
run lists them in a `[warn]` line. Keep other files out of this directory.

## Out-of-target group

The four out-of-target tasks go into a directory of their own, for example to
measure the out-of-target UCE of a UCE-filter subset:

```bash
python benchmarks/preprocess_csr7.py \
    --tasks lambada_openai,commonsense_qa,boolq,social_iqa \
    --output_dir /data/oot4/
```

Pass `/data/oot4/benchmark_group_en/` as `--bench_path`.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--output_dir` | required | Folder for the per-task folders and `benchmark_group_en/` |
| `--tasks` | all 11 | Comma-separated task names from the table above |
| `--no_combined` | off | Write only the per-task files; `benchmark_group_en/` is neither created nor changed |

Exit status: 0 when every task was written; 1 when a task failed, for example
on a download error or a missing split (the other tasks are still written, and
the last line names the failed ones); 2 for an unknown task name (nothing is
written). After a failure, rerun with the same `--tasks`: each run replaces
`benchmark_group_en/`, so a rerun of only the failed task would leave only
that task there.

## Text format

Each row is one example. The `text` field holds the raw Hugging Face fields
listed above, joined by spaces in that order: for most tasks the question or
context followed by all answer options. The punctuation characters
`. ? ! , ; : ( ) [ ] { }` are then bracketed by newlines. The word tokenizer of `unigram_ce` and
`ce_filter` turns punctuation into spaces, so these newlines do not change the
words or the UCE. The text is not the prompt an evaluation harness shows the
model: there is no prompt template and no answer label, and markup such as
HellaSwag's `[header]` is kept.

## Dependencies

- `datasets >= 2.14` (tested with 2.14.7, 3.6.0 and 4.5.0 on Python 3.9 and
  with 5.1.0 on Python 3.14; all four write identical text)
- `pyarrow >= 14`
- No Hugging Face login is needed.

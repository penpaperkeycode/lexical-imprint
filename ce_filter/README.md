# CE-filter — data selection by ΔH (paper §5.1)

A 6-step pipeline that:
1. Scans benchmark text → builds target unigram distribution Q.
2. Scans the pretraining corpus → builds source unigram counts and total
   token count.
3. Computes the row-removal contribution score for every row, in closed
   form (analytical ΔH, O(unique words in row) per call).
4. Keeps the top *p%* of rows by score.
5. Writes the filtered corpus as sharded parquet files.
6. Re-computes UCE on the filtered output and prints the change vs. the
   unfiltered baseline (sanity check).

## Quick example

```bash
# Filter the top 20% of a parquet pretraining corpus by CE-ΔH:
python ce_filter.py \
    --bench_path /data/benchmarks/benchmark_group_en/ \
    --bench_col  text \
    --train_path /data/cc_dump_2026_04/ \
    --train_col  text \
    --train_format parquet \
    --top_pct      20 \
    --out_parquet  /data/filtered_top20/ \
    --out_num_shards 12 \
    --n_workers   120 \
    --batch_size  8192 \
    --worker_batch 1024 \
    --resume
```

The `benchmark_group_en/` directory should contain one parquet per
benchmark task (see `../benchmarks/preprocess_csr7.py`). Per-row text is
concatenated across the directory to form Q.

`--train_format` may be `parquet` (default) or `wet` (for raw Common
Crawl WET files, decoded via `warcio`).

## Resume / checkpointing

After step 3 / 4 / 5 the script writes a pickle to `{out_parquet}.ckpt/`.
With `--resume`, restarting the script picks up where the previous run
stopped. This is useful when scanning very large corpora that exceed
single-job wall-clock limits.

## Verifying CE on the filtered output

```bash
python ce_filter.py \
    --bench_path /data/benchmarks/benchmark_group_en/ \
    --train_path  /data/cc_dump_2026_04/ \
    --out_parquet /data/filtered_top20/ \
    --train_col   text \
    --verify_only
```

`--verify_only` rebuilds the benchmark distribution Q from `--bench_path`
and re-computes UCE against the filtered corpus at `--out_parquet`,
skipping the scoring/writing steps. `--train_path` is still required
by the argparser but is unused in this mode.

## How the score is computed

For each row `r`, the score is the analytical change in CE if `r` were
removed:

    ΔH(r) = log(D_new / D)
          + sum_{w in row r} q(w) · log( max(c(w),1) / max(c(w)−c_r(w),1) )

where:
- `c(w)` = count of word `w` in pretraining corpus (after Laplace add-1
  on benchmark vocabulary)
- `c_r(w)` = count of `w` within row `r`
- `D = N + B`, `D_new = N − |r| + B_new` (B_new accounts for words `r`
  alone keeps in the smoothed vocabulary)

Words not in row `r` cancel exactly, so the sum is only over the row's
unique words — no full-vocab scan, no Counter copies. This is what
makes 120-core scoring of 100B+ tokens tractable on a single machine.

The full derivation is in `ce_filter.py:608` (function
`compute_delta_h_bits_analytical`) and is referenced from §5.1 of the
paper.

## Dependencies

```
pyarrow >= 14
warcio    (only if --train_format wet)
```

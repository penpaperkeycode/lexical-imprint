#!/usr/bin/env python3
"""Preprocess the 11 benchmarks of the paper's UCE-filter study.

Downloads each dataset from the Hugging Face Hub and writes one parquet file
per benchmark, with a single `text` column that joins the raw text fields of
each example (see benchmarks/README.md).

Tasks (paper §5.2):

CSR7 (CE-filter targets):
  - ARC-E         (allenai/ai2_arc, ARC-Easy)
  - ARC-C         (allenai/ai2_arc, ARC-Challenge)
  - HellaSwag     (Rowan/hellaswag)
  - SciQ          (allenai/sciq)
  - OpenBookQA    (allenai/openbookqa, main)
  - PIQA          (ybisk/piqa, Hub parquet conversion refs/convert/parquet)
  - SWAG          (allenai/swag, regular)

Out-of-target (OOT-4):
  - LAMBADA       (EleutherAI/lambada_openai)
  - CommonsenseQA (tau/commonsense_qa)
  - BoolQ         (google/boolq)
  - Social-IQA    (lighteval/siqa)

Output (under --output_dir):
  <output_dir>/<task>/<task>.parquet     (per-task parquet)
  <output_dir>/benchmark_group_en/        (combined directory: the tasks
                                          written by this run)

Each parquet has one `text` column where each row is one example's text (for
most tasks the question or context followed by all answer options; SciQ adds
the support passage, BoolQ is the passage and the question, and LAMBADA is the
passage), ready for unigram-frequency analysis or CE-filter benchmark scan.
A directory passed as --bench_path is pooled into one Q, so build the CSR7
target group (the paper's UCE-filter target) with --tasks and pass its
benchmark_group_en/. Each run removes task files of earlier runs from
benchmark_group_en/ and warns about any other file there that --bench_path
would read.

Usage (from the repository root):
    python benchmarks/preprocess_csr7.py --output_dir /data/benchmarks/

CSR7 target group:
    python benchmarks/preprocess_csr7.py --output_dir /data/csr7/ \
        --tasks arc_easy,arc_challenge,hellaswag,sciq,openbookqa,piqa,swag

Exit status: 0 = all tasks written; 1 = at least one task failed, for example
on a download error or a missing split (the other tasks are still written;
rerun with the same --tasks); 2 = unknown task name.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Callable, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


PUNCT_PATTERN = re.compile(r"([.?!,;:()\[\]{}])")


def sub_punct(text: str) -> str:
    """Insert newlines around punctuation. Matches the paper's preprocessing."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return PUNCT_PATTERN.sub(r"\n\1\n", text)


def save_parquet(rows: List[str], out_path: str, compression: str = "zstd") -> None:
    """Write a list of strings as a single-column parquet (`text`)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = pa.array([x if x is not None else "" for x in rows], type=pa.string())
    table = pa.table({"text": arr})
    pq.write_table(
        table,
        out_path,
        compression=compression,
        use_dictionary=True,
        data_page_version="2.0",
    )
    print(f"  [write] {len(arr):,} rows → {out_path}", flush=True)


def out_path(output_dir: str, task: str) -> str:
    return os.path.join(output_dir, task, f"{task}.parquet")


# ----------------------------------------------------------------------
# Per-task processors. Each takes load_dataset args, extracts text, returns list.
# ----------------------------------------------------------------------

def proc_arc_easy() -> List[str]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            choices = " ".join(ex["choices"]["text"])
            rows.append(sub_punct(f"{ex['question']} {choices}"))
    return rows


def proc_arc_challenge() -> List[str]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            choices = " ".join(ex["choices"]["text"])
            rows.append(sub_punct(f"{ex['question']} {choices}"))
    return rows


def proc_hellaswag() -> List[str]:
    ds = load_dataset("Rowan/hellaswag")
    rows = []
    for split in ("train", "validation"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            ctx = ex.get("ctx", "") or ""
            endings = " ".join(ex.get("endings", []))
            rows.append(sub_punct(f"{ctx} {endings}"))
    return rows


def proc_sciq() -> List[str]:
    ds = load_dataset("allenai/sciq")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            choices = " ".join(
                str(ex.get(k, "") or "")
                for k in ("correct_answer", "distractor1", "distractor2", "distractor3")
            )
            support = ex.get("support", "") or ""
            rows.append(sub_punct(f"{ex['question']} {choices} {support}"))
    return rows


def proc_openbookqa() -> List[str]:
    ds = load_dataset("allenai/openbookqa", "main")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            stem = ex["question_stem"]
            choices = " ".join(ex["choices"]["text"])
            rows.append(sub_punct(f"{stem} {choices}"))
    return rows


def proc_piqa() -> List[str]:
    # The main branch of ybisk/piqa holds only a loading script, which datasets
    # >= 4.0 no longer runs; read the Hub's parquet conversion instead.
    ds = load_dataset("ybisk/piqa", revision="refs/convert/parquet")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            rows.append(sub_punct(f"{ex['goal']} {ex['sol1']} {ex['sol2']}"))
    return rows


def proc_swag() -> List[str]:
    ds = load_dataset("allenai/swag", "regular")
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            startphrase = ex.get("startphrase", "") or ""
            endings = " ".join(
                str(ex.get(k, "") or "") for k in ("ending0", "ending1", "ending2", "ending3")
            )
            rows.append(sub_punct(f"{startphrase} {endings}"))
    return rows


def proc_lambada() -> List[str]:
    ds = load_dataset("EleutherAI/lambada_openai", "en")
    rows = []
    for split in ("test",):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            rows.append(sub_punct(ex["text"]))
    return rows


def proc_commonsense_qa() -> List[str]:
    ds = load_dataset("tau/commonsense_qa")
    rows = []
    for split in ("train", "validation"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            choices = " ".join(ex["choices"]["text"])
            rows.append(sub_punct(f"{ex['question']} {choices}"))
    return rows


def proc_boolq() -> List[str]:
    ds = load_dataset("google/boolq")
    rows = []
    for split in ("train", "validation"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            rows.append(sub_punct(f"{ex.get('passage', '')} {ex.get('question', '')}"))
    return rows


def proc_social_iqa() -> List[str]:
    ds = load_dataset("lighteval/siqa")
    rows = []
    for split in ("train", "validation"):
        if split not in ds:
            raise ValueError(f"split '{split}' not found; splits present: {list(ds)}")
        for ex in ds[split]:
            ctx = ex.get("context", "") or ""
            q = ex.get("question", "") or ""
            choices = " ".join(
                str(ex.get(k, "") or "") for k in ("answerA", "answerB", "answerC")
            )
            rows.append(sub_punct(f"{ctx} {q} {choices}"))
    return rows


TASK_PROCESSORS: Dict[str, Callable[[], List[str]]] = {
    # CSR7 (CE-filter targets in the paper)
    "arc_easy":       proc_arc_easy,
    "arc_challenge":  proc_arc_challenge,
    "hellaswag":      proc_hellaswag,
    "sciq":           proc_sciq,
    "openbookqa":     proc_openbookqa,
    "piqa":           proc_piqa,
    "swag":           proc_swag,
    # Out-of-target (OOT-4)
    "lambada_openai": proc_lambada,
    "commonsense_qa": proc_commonsense_qa,
    "boolq":          proc_boolq,
    "social_iqa":     proc_social_iqa,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Directory under which per-task subdirectories are created")
    ap.add_argument("--tasks", default="",
                    help=f"Comma-separated subset; default = all 11. "
                         f"Choices: {','.join(TASK_PROCESSORS.keys())}")
    ap.add_argument("--no_combined", action="store_true",
                    help="Do not create or touch the combined benchmark_group_en/ directory")
    args = ap.parse_args()

    selected = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(TASK_PROCESSORS)
    unknown = [t for t in selected if t not in TASK_PROCESSORS]
    if unknown:
        print(f"[error] unknown tasks: {unknown}", file=sys.stderr)
        sys.exit(2)

    print(f"[preprocess] processing {len(selected)} task(s): {selected}", flush=True)
    combined_dir = os.path.join(args.output_dir, "benchmark_group_en")
    if not args.no_combined:
        # --bench_path pools every parquet file in the combined directory into one Q,
        # so it must hold only this run's tasks: remove task files left by earlier
        # runs (files with other names are left alone).
        removed = []
        for task in TASK_PROCESSORS:
            stale = os.path.join(combined_dir, f"{task}.parquet")
            if os.path.isfile(stale):
                os.remove(stale)
                removed.append(os.path.basename(stale))
        if removed:
            print(f"[combined] removed {len(removed)} file(s) of an earlier run from "
                  f"{combined_dir}: {removed}", flush=True)

    failed = []
    for task in selected:
        print(f"\n[process] {task}", flush=True)
        try:
            rows = TASK_PROCESSORS[task]()
            if not rows:
                raise ValueError("no rows (the splits read are empty)")
        except Exception as e:
            print(f"  [FAIL] {task}: {e}", file=sys.stderr)
            failed.append(task)
            continue
        per_task = out_path(args.output_dir, task)
        save_parquet(rows, per_task)
        if not args.no_combined:
            save_parquet(rows, os.path.join(combined_dir, f"{task}.parquet"))

    if not args.no_combined and os.path.isdir(combined_dir):
        # Files this run did not write are kept but would also be pooled into Q
        # (--bench_path skips only names starting with '.' or '_'), so name them.
        written = {f"{t}.parquet" for t in selected if t not in failed}
        others = sorted(f for f in os.listdir(combined_dir)
                        if f not in written and not f.startswith((".", "_")))
        if others:
            print(f"\n[warn] {combined_dir} also holds files not written by this run, "
                  f"and --bench_path pools them too: {others}", file=sys.stderr)

    if failed:
        hint = "" if args.no_combined else (
            "; rerun with the same --tasks, since each run replaces benchmark_group_en/")
        print(f"\n[error] {len(failed)} of {len(selected)} task(s) failed: {failed}; "
              f"{len(selected) - len(failed)} written under {args.output_dir}{hint}",
              file=sys.stderr)
        sys.exit(1)
    print(f"\n[done] outputs at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

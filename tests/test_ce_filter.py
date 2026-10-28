"""Tests for the UCE-filter (ce_filter/ce_filter.py, paper Section 5.1).

Toy data in tmp_path, no network. CLI tests run the script in a subprocess with 2 workers.
The reference selection below recomputes UCE from scratch for every row (no analytic ΔH).
"""
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ce_filter.ce_filter import compute_delta_h_bits_analytical, tokenize

SCRIPT = Path(__file__).resolve().parents[1] / "ce_filter" / "ce_filter.py"
WORKERS = ["--n_workers", "2", "--worker_batch", "3", "--batch_size", "4"]

BENCH_A = [
    "The cat sat on the mat.",
    "A dog ran in the park, then the dog slept.",
    "The sun is hot today; birds fly south in winter.",
]
BENCH_B = ["Rivers flow to the sea and boats sail on rivers."]
WORDS = ("The cat sat on the mat A dog ran in park sun is hot today birds fly south winter "
         "then slept zz yy xx qq ww vv uu tt").split()


# ------------------------------------------------------------------
# Reference (brute force, Appendix A.1)
# ------------------------------------------------------------------
def ref_uce_bits(q, counts, total_tokens):
    """-sum_w q(w) log2(max(c(w), 1) / (T + B)); B = number of benchmark words with c(w) = 0."""
    B = sum(1 for w in q if counts.get(w, 0) == 0)
    D = total_tokens + B
    return -sum(qw * math.log2(max(counts.get(w, 0), 1) / D) for w, qw in q.items())


def ref_q(bench_texts):
    """Token-weighted Q pooled over all benchmark rows."""
    c = Counter()
    for t in bench_texts:
        c.update(tokenize(t))
    n = sum(c.values())
    return {w: v / n for w, v in c.items()}


def ref_removed(bench_texts, corpus_texts, top_pct):
    """Physical row indices the documented rule removes, from brute-force ΔH.

    K = max(1, int(N_text * top_pct / 100)) (0 for top_pct 0) over the rows with text;
    rows are removed in order of (ΔH ascending, row index descending); null rows are kept.
    """
    q = ref_q(bench_texts)
    toks = {i: tokenize(t) for i, t in enumerate(corpus_texts) if t is not None}
    counts, total = Counter(), 0
    for tk in toks.values():
        counts.update(w for w in tk if w in q)
        total += len(tk)
    h0 = ref_uce_bits(q, counts, total)
    dh = {}
    for i, tk in toks.items():
        c2 = counts.copy()
        c2.subtract(w for w in tk if w in q)
        dh[i] = ref_uce_bits(q, c2, total - len(tk)) - h0
    n_text = len(toks)
    k = 0 if top_pct == 0 else min(n_text, max(1, int(n_text * top_pct / 100.0)))
    order = sorted(dh, key=lambda i: (dh[i], -i))
    if 0 < k < n_text:
        # The toy data must not have near-ties at the cutoff (exact ties are fine)
        lo, hi = dh[order[k - 1]], dh[order[k]]
        assert lo == hi or hi - lo > 1e-9, (lo, hi)
    return set(order[:k]), dh, order, k


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def write_bench(d, groups):
    """groups: {file_stem: [texts]} -> one parquet per stem with a `text` column."""
    os.makedirs(d, exist_ok=True)
    for stem, texts in groups.items():
        pq.write_table(pa.table({"text": texts}), os.path.join(d, f"{stem}.parquet"))
    return d


def write_corpus(d, texts, n_files=2, row_group_size=5):
    """Corpus over n_files parquet files with columns text (nullable) and rid (physical row id)."""
    os.makedirs(d, exist_ok=True)
    per = math.ceil(len(texts) / n_files)
    for f in range(n_files):
        lo, hi = f * per, min(len(texts), (f + 1) * per)
        t = pa.table({"text": pa.array(texts[lo:hi], type=pa.string()),
                      "rid": pa.array(range(lo, hi), type=pa.int64())})
        pq.write_table(t, os.path.join(d, f"part-{f:02d}.parquet"), row_group_size=row_group_size)
    return d


def random_texts(n, seed):
    """n rows with distinct token multisets (so equal ΔH only comes from equal content)."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        toks = [rng.choice(WORDS) for _ in range(rng.randint(2, 14))]
        key = tuple(sorted(toks))
        if key not in seen:
            seen.add(key)
            out.append(" ".join(toks) + rng.choice([".", "!", ""]))
    return out


def run_cli(*args, cwd=None, temp_dir=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if temp_dir is not None:
        env["TMPDIR"] = str(temp_dir)
    return subprocess.run([sys.executable, str(SCRIPT), *[str(a) for a in args]],
                          capture_output=True, text=True, env=env, cwd=cwd, timeout=300)


def kept_rids(path):
    return pq.read_table(path).column("rid").to_pylist()


def error_lines(r):
    return [line for line in r.stderr.splitlines() if line.startswith("[error]")]


def edit_pickle(src, fn, dest=None):
    """Load a checkpoint pickle, apply fn to its dict, write it to dest (default: in place)."""
    with open(src, "rb") as f:
        d = pickle.load(f)
    fn(d)
    with open(dest or src, "wb") as f:
        pickle.dump(d, f)


# ------------------------------------------------------------------
# Analytic ΔH
# ------------------------------------------------------------------
def test_analytic_delta_h_equals_brute_force():
    rng = random.Random(7)
    n_rows = n_b_changes = 0
    for trial in range(30):
        q = ref_q(rng.sample(BENCH_A + BENCH_B, rng.randint(1, 4)))
        rows = random_texts(rng.randint(3, 15), seed=trial)
        # empty row, no benchmark word, benchmark words that occur only in that row
        rows += ["", "zz yy", "Rivers boats", "sea sea sail!", "flow"]
        toks = [tokenize(t) for t in rows]
        counts, total = Counter(), 0
        for tk in toks:
            counts.update(w for w in tk if w in q)
            total += len(tk)
        B = sum(1 for w in q if counts.get(w, 0) <= 0)
        D = float(total + B)
        h0 = ref_uce_bits(q, counts, total)
        for tk in toks:
            row_counter = Counter(w for w in tk if w in q)
            got = compute_delta_h_bits_analytical(row_counter, len(tk), q, counts, total, B, D)
            c2 = counts.copy()
            c2.subtract(row_counter)
            want = ref_uce_bits(q, c2, total - len(tk)) - h0
            assert abs(got - want) < 1e-12, (tk, got, want)
            n_rows += 1
            n_b_changes += any(counts[w] == c for w, c in row_counter.items())
    assert n_rows > 300
    assert n_b_changes > 60  # rows whose removal makes a benchmark word unseen (B changes)


# ------------------------------------------------------------------
# --top_pct semantics
# ------------------------------------------------------------------
def test_top_pct_80_removes_80_percent_and_keeps_brute_force_rows(tmp_path):
    texts = random_texts(50, seed=1)
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A, "b": BENCH_B})
    corpus = write_corpus(tmp_path / "corpus", texts)
    out = tmp_path / "out.parquet"
    r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                "--top_pct", "80", "--skip_seq_apply", *WORKERS)
    assert r.returncode == 0, r.stderr
    removed, _, _, k = ref_removed(BENCH_A + BENCH_B, texts, 80)
    assert k == int(0.8 * 50)
    assert kept_rids(out) == [i for i in range(50) if i not in removed]  # N - int(0.8 N) rows, in order
    assert "rows removed = 40 / 50 (80.0%)" in r.stdout


def test_default_top_pct_is_the_paper_setting(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", random_texts(10, seed=11))
    out = tmp_path / "out.parquet"
    r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                "--skip_step6", *WORKERS)
    assert r.returncode == 0, r.stderr
    assert "top_pct=80.0% | K=8" in r.stderr
    assert len(kept_rids(out)) == 2


def test_top_pct_outside_range_exits_nonzero(tmp_path):
    for bad in ("-1", "100", "150", "nan"):
        r = run_cli("--bench_path", tmp_path, "--train_path", tmp_path,
                    "--out_parquet", tmp_path / "o.parquet", f"--top_pct={bad}")
        assert r.returncode != 0
        assert "--top_pct must be in [0, 100)" in r.stderr


def test_top_pct_zero_keeps_every_row(tmp_path):
    texts = random_texts(20, seed=3)
    texts[4] = None
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", texts)
    out = tmp_path / "out.parquet"
    r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                "--top_pct", "0", "--skip_step6", *WORKERS)
    assert r.returncode == 0, r.stderr
    assert kept_rids(out) == list(range(20))
    assert "rows removed = 0 / 20 (0.0%)" in r.stdout


def test_fractional_top_pct_warns(tmp_path):
    texts = random_texts(20, seed=4)
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", texts)
    out = tmp_path / "out.parquet"
    r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                "--top_pct", "0.8", "--skip_step6", *WORKERS)
    assert r.returncode == 0, r.stderr
    assert "[warn] --top_pct 0.8" in r.stderr
    assert len(kept_rids(out)) == 19  # K = max(1, int(20 * 0.8 / 100)) = 1


# ------------------------------------------------------------------
# --n_workers
# ------------------------------------------------------------------
def test_n_workers_must_be_positive_and_is_capped_at_the_cpu_count(tmp_path):
    r = run_cli("--bench_path", tmp_path, "--train_path", tmp_path,
                "--out_parquet", tmp_path / "o.parquet", "--n_workers", "0")
    assert r.returncode == 2 and "--n_workers must be at least 1" in r.stderr
    # An empty benchmark directory stops the run in Step 1 before any pool starts
    empty = tmp_path / "empty_bench"
    empty.mkdir()
    r = run_cli("--bench_path", empty, "--out_parquet", tmp_path / "o.parquet",
                "--verify_only", "--n_workers", "1000000")
    cpus = os.cpu_count() or 1
    assert f"[config] --n_workers 1000000 capped at the CPU count: {cpus}" in r.stderr
    assert f"[config] n_workers={cpus} |" in r.stderr
    assert r.returncode == 1 and error_lines(r) == [
        f"[error] ValueError: No fragments found in {empty}. Check path and format."]


# ------------------------------------------------------------------
# Null text rows
# ------------------------------------------------------------------
NULL_POS = [0, 13, 19, 20, 27, 28, 29, 39]  # first, middle, file boundary, consecutive, last


@pytest.fixture(scope="module")
def null_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nulls")
    texts = random_texts(40, seed=2)
    texts[5] = texts[33] = ""                   # empty strings are ordinary rows (ΔH = 0)
    for i in NULL_POS:
        texts[i] = None
    bench = write_bench(tmp / "bench", {"a": BENCH_A, "b": BENCH_B})
    corpus = write_corpus(tmp / "corpus", texts)
    out = tmp / "out.parquet"
    # seq-apply check on (no --skip_seq_apply)
    r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                "--top_pct", "80", *WORKERS)
    return texts, bench, corpus, out, r


def test_null_rows_are_kept_and_do_not_shift_the_selection(null_run):
    texts, _, _, out, r = null_run
    assert r.returncode == 0, r.stderr
    removed, _, _, k = ref_removed(BENCH_A + BENCH_B, texts, 80)
    n_text = 40 - len(NULL_POS)
    assert k == int(0.8 * n_text)               # K counts only rows with text
    assert not removed & set(NULL_POS)
    assert kept_rids(out) == [i for i in range(40) if i not in removed]   # includes every null row
    assert f"null rows    = {len(NULL_POS)}" in r.stdout
    assert "H_seq_apply - H_after = 0.000000e+00 bits" in r.stdout


def test_sharded_output_holds_the_rows_of_the_single_file_output(null_run, tmp_path):
    _, bench, corpus, out, r = null_run
    shards = tmp_path / "shards"
    # --batch_size only changes the read batches; with 64 a batch is split across two shards
    r3 = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", shards,
                 "--top_pct", "80", "--out_num_shards", "3", *WORKERS, "--batch_size", "64")
    assert r3.returncode == 0, r3.stderr
    files = sorted(shards.iterdir())
    assert [f.name for f in files] == ["part-0000.parquet", "part-0001.parquet", "part-0002.parquet"]
    assert [x for f in files for x in kept_rids(f)] == kept_rids(out)   # same rows, same order
    assert r3.stdout == r.stdout                # same H values, counts and seq-apply check


def test_verify_only_needs_no_train_path(null_run):
    _, bench, _, out, r = null_run
    h_after = next(l for l in r.stdout.splitlines() if l.startswith("[filtered]")).split("=")[1]
    # --verify_only does not use --top_pct, so it skips the range check
    v = run_cli("--bench_path", bench, "--out_parquet", out, "--verify_only", "--top_pct", "100",
                *WORKERS)
    assert v.returncode == 0, v.stderr
    assert f"H_after (recomputed) = {h_after.strip()}" in v.stdout


# ------------------------------------------------------------------
# Ties at the cutoff
# ------------------------------------------------------------------
def test_ties_at_cutoff_are_broken_by_row_index(tmp_path):
    rng = random.Random(5)
    aligned = random_texts(12, seed=6)
    junk = [f"j{i}a j{i}b j{i}c" for i in range(24)]   # same length, no benchmark word: equal ΔH
    texts = aligned + junk
    rng.shuffle(texts)
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", texts, n_files=3, row_group_size=4)
    removed, dh, order, k = ref_removed(BENCH_A, texts, 50)
    assert dh[order[k - 1]] == dh[order[k]]            # the cutoff falls inside the tie
    want = [i for i in range(len(texts)) if i not in removed]
    outputs = []
    for rep in range(3):
        out = tmp_path / f"out{rep}.parquet"
        r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out,
                    "--top_pct", "50", "--skip_step6", "--n_workers", "2", "--worker_batch", "1")
        assert r.returncode == 0, r.stderr
        outputs.append(kept_rids(out))
    assert outputs[0] == outputs[1] == outputs[2] == want


# ------------------------------------------------------------------
# Failures that used to pass silently
# ------------------------------------------------------------------
def test_benchmark_file_without_text_column_exits_nonzero(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    pq.write_table(pa.table({"question": BENCH_B}), tmp_path / "bench" / "b_question.parquet")
    # small good files after the bad one: workers still write pickles while the error travels
    write_bench(tmp_path / "bench", {f"c{i:02d}": BENCH_A[:1] for i in range(20)})
    corpus = write_corpus(tmp_path / "corpus", random_texts(10, seed=8))
    temp = tmp_path / "temp"
    temp.mkdir()
    r = run_cli("--bench_path", bench, "--train_path", corpus,
                "--out_parquet", tmp_path / "out.parquet", *WORKERS, temp_dir=temp)
    assert r.returncode != 0
    assert "b_question.parquet" in r.stderr and "no column 'text'" in r.stderr
    assert not (tmp_path / "out.parquet").exists()
    assert os.listdir(temp) == []               # no temp pickles left behind


def test_corpus_file_without_text_column_exits_nonzero(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", random_texts(10, seed=9))
    pq.write_table(pa.table({"body": ["the cat sat"], "rid": [10]}),
                   tmp_path / "corpus" / "part-00-nocol.parquet")   # listed first
    for i in range(20):
        pq.write_table(pa.table({"text": ["the cat sat"], "rid": [11 + i]}),
                       tmp_path / "corpus" / f"part-{10 + i}.parquet")
    temp = tmp_path / "temp"
    temp.mkdir()
    r = run_cli("--bench_path", bench, "--train_path", corpus,
                "--out_parquet", tmp_path / "out.parquet", *WORKERS, temp_dir=temp)
    assert r.returncode != 0
    assert "part-00-nocol.parquet" in r.stderr and "no column 'text'" in r.stderr
    assert os.listdir(temp) == []               # no temp pickles left behind


def test_corpus_without_word_tokens_exits_nonzero(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    for name, texts in (("all_null", [None] * 6),
                        ("no_words", ["", "!!!", "...", None, " - ", "?"])):
        corpus = write_corpus(tmp_path / name, texts)
        out = tmp_path / f"{name}.parquet"
        r = run_cli("--bench_path", bench, "--train_path", corpus, "--out_parquet", out, *WORKERS)
        assert r.returncode == 1
        assert error_lines(r) == [f"[error] train-scan(before): no word tokens in column 'text' "
                                  f"of pretraining corpus at {corpus}"]
        assert "H_before" not in r.stdout and not out.exists()


def test_unreadable_parquet_file_gives_one_error_line(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp_path / "corpus", random_texts(10, seed=13))
    bad_bench = write_bench(tmp_path / "bad_bench", {"b": BENCH_B})
    (bad_bench / "a_garbage.parquet").write_bytes(b"not a parquet file\n" * 20)   # listed first
    bad_corpus = write_corpus(tmp_path / "bad_corpus", random_texts(10, seed=13))
    (bad_corpus / "part-02.parquet").write_bytes(b"\x00garbage" * 40)              # listed last
    for b, c, bad in ((bad_bench, corpus, bad_bench / "a_garbage.parquet"),
                      (bench, bad_corpus, bad_corpus / "part-02.parquet")):
        r = run_cli("--bench_path", b, "--train_path", c,
                    "--out_parquet", tmp_path / "out.parquet", *WORKERS)
        assert r.returncode == 1
        errors = error_lines(r)
        assert len(errors) == 1 and f"'{bad}'" in errors[0], r.stderr
        assert "Traceback" not in r.stderr
    # --verify_only on an output that does not exist
    r = run_cli("--bench_path", bench, "--out_parquet", tmp_path / "never.parquet",
                "--verify_only", *WORKERS)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert error_lines(r) == [f"[error] FileNotFoundError: {tmp_path / 'never.parquet'}"]


def test_broken_symlink_and_symlink_loop_exit_nonzero(tmp_path):
    bench = write_bench(tmp_path / "bench", {"a": BENCH_A})
    broken = write_corpus(tmp_path / "broken", random_texts(10, seed=14))
    os.symlink(tmp_path / "missing.parquet", broken / "part-02.parquet")
    loop = write_corpus(tmp_path / "loop", random_texts(10, seed=14))
    os.symlink(".", loop / "self")
    for corpus, want in (
        (broken, f"[error] FileNotFoundError: broken symlink: {broken / 'part-02.parquet'}"),
        (loop, f"[error] ValueError: {loop}: the same file is reached through more than one "
               f"path (symlink loop?)"),
    ):
        r = run_cli("--bench_path", bench, "--train_path", corpus,
                    "--out_parquet", tmp_path / "out.parquet", *WORKERS)
        assert r.returncode == 1
        assert error_lines(r) == [want], r.stderr


def test_directories_of_valid_symlinks_still_run(tmp_path):
    texts = random_texts(20, seed=15)
    real_corpus = write_corpus(tmp_path / "real_corpus", texts)
    real_bench = write_bench(tmp_path / "real_bench", {"a": BENCH_A, "b": BENCH_B})
    for real, links in ((real_corpus, tmp_path / "corpus"), (real_bench, tmp_path / "bench")):
        links.mkdir()
        for f in sorted(real.iterdir()):
            os.symlink(f, links / f.name)
    out = tmp_path / "out.parquet"
    r = run_cli("--bench_path", tmp_path / "bench", "--train_path", tmp_path / "corpus",
                "--out_parquet", out, "--top_pct", "80", "--skip_step6", *WORKERS)
    assert r.returncode == 0, r.stderr
    removed, _, _, _ = ref_removed(BENCH_A + BENCH_B, texts, 80)
    assert kept_rids(out) == [i for i in range(20) if i not in removed]


def test_resume_refuses_changed_settings_and_resumes_matching_ones(tmp_path):
    texts = random_texts(30, seed=10)
    bench_a = write_bench(tmp_path / "bench_a", {"a": BENCH_A})
    bench_b = write_bench(tmp_path / "bench_b", {"b": BENCH_B})
    corpus = write_corpus(tmp_path / "corpus", texts)
    out = tmp_path / "out.parquet"
    out.mkdir()   # a directory where the single output file goes: Step 5 fails after Steps 3-4
    base = ["--bench_path", bench_a, "--train_path", corpus, "--out_parquet", out,
            "--skip_seq_apply", "--resume", *WORKERS]
    r = run_cli(*base, "--top_pct", "60")
    assert r.returncode != 0
    ckpt = tmp_path / "out.parquet.ckpt"
    assert (ckpt / "step4.pkl").exists() and (ckpt / "run_config.json").exists()

    r = run_cli(*base, "--top_pct", "80")
    assert r.returncode != 0 and "--resume" in r.stderr and "top_pct: 60.0 -> 80.0" in r.stderr
    changed_bench = [a if a != bench_a else bench_b for a in base]
    r = run_cli(*changed_bench, "--top_pct", "60")
    assert r.returncode != 0 and "--resume" in r.stderr and "bench_path" in r.stderr

    out.rmdir()
    r = run_cli(*base, "--top_pct", "60")
    assert r.returncode == 0, r.stderr
    assert "Resuming from Step 5" in r.stderr
    removed, _, _, _ = ref_removed(BENCH_A, texts, 60)
    assert kept_rids(out) == [i for i in range(30) if i not in removed]
    assert not ckpt.exists()


# ------------------------------------------------------------------
# Safety nets, triggered by edited checkpoints and --resume
# ------------------------------------------------------------------
GUARD_NULLS = [7, 30]


@pytest.fixture(scope="module")
def blocked_run(tmp_path_factory):
    """Steps 1-4 of one run whose Step 5 is blocked; each case copies its checkpoints."""
    tmp = tmp_path_factory.mktemp("guards")
    texts = random_texts(40, seed=12)
    for i in GUARD_NULLS:
        texts[i] = None
    bench = write_bench(tmp / "bench", {"a": BENCH_A})
    corpus = write_corpus(tmp / "corpus", texts)
    args = ["--bench_path", bench, "--train_path", corpus, "--top_pct", "50", "--resume", *WORKERS]
    out = tmp / "out.parquet"
    out.mkdir()   # a directory where the single output file goes: Step 5 fails after Steps 3-4
    r = run_cli(*args, "--out_parquet", out, "--skip_seq_apply")
    assert r.returncode == 1, r.stderr
    ckpt = tmp / "out.parquet.ckpt"
    assert (ckpt / "step3.pkl").exists() and (ckpt / "step4.pkl").exists()
    return texts, corpus, args, ckpt


def resume_case(blocked_run, tmp_path, edit, *extra):
    """Copy the blocked run's checkpoints for a fresh output path, edit them, rerun with --resume."""
    _, _, args, ckpt = blocked_run
    out = tmp_path / "out.parquet"
    if not (tmp_path / "out.parquet.ckpt").exists():
        shutil.copytree(ckpt, tmp_path / "out.parquet.ckpt")
    edit(tmp_path / "out.parquet.ckpt")
    return run_cli(*args, "--out_parquet", out, *extra), out


def test_scoring_guard_stops_on_a_null_count_mismatch(blocked_run, tmp_path):
    def edit(ck):
        os.remove(ck / "step4.pkl")             # resume at Step 4 from the step3 checkpoint
        edit_pickle(ck / "step3.pkl", lambda d: d.update(null_rows=d["null_rows"] + 3))
    r, _ = resume_case(blocked_run, tmp_path, edit, "--skip_seq_apply")
    assert r.returncode == 1
    errors = error_lines(r)
    assert len(errors) == 1 and errors[0].startswith("[error] deltaH-score: 2 null rows"), r.stderr


def test_write_guard_stops_when_the_written_rows_do_not_add_up(blocked_run, tmp_path):
    def out_of_range(d):
        d["remove_indices"].discard(max(d["remove_indices"]))
        d["remove_indices"].add(10 ** 6)
    r, out = resume_case(blocked_run, tmp_path,
                         lambda ck: edit_pickle(ck / "step4.pkl", out_of_range), "--skip_seq_apply")
    assert r.returncode == 1
    assert error_lines(r) == [f"[error] write: read 40 of 40 corpus rows and wrote 22 rows, "
                              f"expected 21; {out} is not valid"]


def test_seq_apply_guard_stops_when_the_output_is_not_the_selection(blocked_run, tmp_path):
    texts = blocked_run[0]
    # 1. write the correct output (resume after Step 4, no Step 6)
    r, out = resume_case(blocked_run, tmp_path, lambda ck: None, "--skip_step6")
    assert r.returncode == 0 and out.exists(), r.stderr

    # 2. a step5 checkpoint whose removal set has one removed row swapped for a kept row
    def swap(d):
        removed = d["remove_indices"]
        gone = min(removed)
        kept = next(i for i, t in enumerate(texts) if t is not None and i not in removed
                    and len(tokenize(t)) != len(tokenize(texts[gone])))
        removed.discard(gone)
        removed.add(kept)
    r, _ = resume_case(blocked_run, tmp_path,
                       lambda ck: edit_pickle(ck / "step4.pkl", swap, dest=ck / "step5.pkl"))
    assert r.returncode == 1
    assert "Resuming from Step 6" in r.stderr
    errors = error_lines(r)
    assert len(errors) == 1 and errors[0].startswith("[error] seq-apply: |H_seq_apply - H_after|")


def test_resume_refuses_a_changed_corpus_file(blocked_run, tmp_path):
    corpus = blocked_run[1]
    f = sorted(Path(corpus).iterdir())[0]
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    try:
        r, _ = resume_case(blocked_run, tmp_path, lambda ck: None, "--skip_seq_apply")
    finally:
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert r.returncode == 1
    errors = error_lines(r)
    assert len(errors) == 1 and errors[0].startswith("[error] --resume:"), r.stderr
    assert "corpus_files: file list, sizes, mtimes or row count changed" in errors[0]

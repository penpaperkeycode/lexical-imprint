"""Tests for the UCE metric (unigram_ce/compute_unigram_ce.py).

Toy parquet data only; no network. CLI tests run the script with sys.executable.
"""
import math
import os
import re
import subprocess
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unigram_ce.compute_unigram_ce import (
    build_benchmark_q,
    build_train_counts_parquet,
    compute_uce_nats,
    list_parquet_files,
    nats_to_bits,
    tokenize,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "unigram_ce", "compute_unigram_ce.py")
UCE_LINE = re.compile(r"^UCE = ([0-9.]+) nats(?:\s+\(([0-9.]+) bits\))?$", re.M)
NOT_PARQUET = b"PAR1" + b"\x00" * 64 + b"PAR1"


def write_parquet(path, texts, col="text"):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    pq.write_table(pa.table({col: pa.array(texts, pa.string())}), str(path))


def run_cli(bench, train, *extra):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, SCRIPT, "--bench_path", str(bench), "--train_path", str(train),
           "--n_workers", "2"] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


def parse_uce(stdout):
    m = UCE_LINE.search(stdout)
    assert m, stdout
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def a1_uce_nats(q, c, T):
    """Appendix A.1 written out: P(w) = max(c(w), 1) / (T + B), B = #{w in V_b : c(w) = 0}."""
    B = sum(1 for w in q if c.get(w, 0) == 0)
    return -sum(qw * math.log(max(c.get(w, 0), 1) / (T + B)) for w, qw in q.items())


# ------------------------------------------------------------
# Formula
# ------------------------------------------------------------
def test_compute_uce_nats_matches_hand_computed_a1(tmp_path):
    # Benchmark "the cat sat" / "the dog": Q = {the: 2/5, cat: 1/5, sat: 1/5, dog: 1/5}.
    # Corpus "the cat the mat" / "a mat": T = 6 word tokens (all words, not only V_b),
    # c(the) = 2, c(cat) = 1, sat and dog unseen -> B = 2, T + B = 8.
    q = {"the": 2 / 5, "cat": 1 / 5, "sat": 1 / 5, "dog": 1 / 5}
    counts = Counter({"the": 2, "cat": 1})
    expected = -(2 / 5 * math.log(2 / 8) + 3 * (1 / 5 * math.log(1 / 8)))
    assert nats_to_bits(expected) == pytest.approx(2.6, abs=1e-12)  # 0.4*2 + 0.6*3 bits
    assert compute_uce_nats(q, counts, 6) == pytest.approx(expected, abs=1e-12)

    # Add-one on V_b, or T counted over V_b only, would give other values.
    add_one = -(2 / 5 * math.log(3 / 10) + 1 / 5 * math.log(2 / 10) + 2 / 5 * math.log(1 / 10))
    t_vb_only = -(2 / 5 * math.log(2 / 5) + 3 * (1 / 5 * math.log(1 / 5)))
    assert abs(add_one - expected) > 0.05 and abs(t_vb_only - expected) > 0.05

    # Same numbers from parquet through the library path.
    write_parquet(tmp_path / "bench.parquet", ["the cat sat", "the dog"])
    write_parquet(tmp_path / "corpus" / "part-0.parquet", ["the cat the mat", "a mat"])
    q2, vocab = build_benchmark_q(str(tmp_path / "bench.parquet"))
    assert q2 == pytest.approx(q) and vocab == set(q)
    counts2, T = build_train_counts_parquet(str(tmp_path / "corpus"), "text", vocab, n_workers=1)
    assert T == 6 and counts2 == counts
    assert compute_uce_nats(q2, counts2, T) == pytest.approx(expected, abs=1e-12)


def test_zero_total_tokens_is_rejected():
    with pytest.raises(ValueError):
        compute_uce_nats({"a": 1.0}, Counter(), 0)


# ------------------------------------------------------------
# Tokenizer shared with the UCE-filter
# ------------------------------------------------------------
TRICKY = [
    "ﬁne ｆｕｌｌ Ⅻ ½ x² ℌ ㎏",  # NFKC: ligature, fullwidth, numerals
    "é é ä ẛ̣",                         # combining marks
    "don't stop—now! (really?) «quoted» U.S.A. e-mail #tag @user",
    "3.14 1,000 0x1F ٣٤٥ ১২",           # ASCII and non-ASCII digits
    "snake_case __init__ _x_ ﹏ ＿ ‿",               # underscores, NFKC forms of _, U+203F
    "Apple apple APPLE ÄÖÜ äöü İstanbul Straße",  # case kept
    "한국어 텍스트. 日本語のテキスト。中文，标点！",
    "Русский, ещё. हिन्दी "
    "العربية עברית ไทย",
    "co­operate zero​width tab\tnew\nline nbsp　ideo",  # format chars, spaces
    "I ❤ NY \U0001f600 ™ © € $ % ^ & * + = < > | ~ `",  # symbols and emoji
    "",
    "   ",
    "...",
]


def test_tokenizer_matches_ce_filter():
    from ce_filter.ce_filter import tokenize as cef_tokenize
    for s in TRICKY:
        assert tokenize(s) == cef_tokenize(s), repr(s)
    # The rule itself: NFKC, every non-[\w\s] character -> space, whitespace split, case kept.
    assert tokenize("Don't U.S.A. 3.14 snake_case ﬁne") == [
        "Don", "t", "U", "S", "A", "3", "14", "snake_case", "fine"]


# ------------------------------------------------------------
# Pooled benchmark directory
# ------------------------------------------------------------
def test_directory_bench_is_token_weighted_pool_not_mean(tmp_path):
    corpus = tmp_path / "corpus"
    write_parquet(corpus / "part-0.parquet", ["x"] * 90 + ["y"] * 9 + ["z"])  # T = 100
    bench = tmp_path / "bench"
    write_parquet(bench / "a.parquet", ["x x x x", "x x x x"])  # 8 tokens
    write_parquet(bench / "b.parquet", ["z y w"])  # 3 tokens; w is unseen in the corpus
    c = {"x": 90, "y": 9, "z": 1}
    pooled = a1_uce_nats({"x": 8 / 11, "y": 1 / 11, "z": 1 / 11, "w": 1 / 11}, c, 100)
    per_file = [a1_uce_nats({"x": 1.0}, c, 100),
                a1_uce_nats({"z": 1 / 3, "y": 1 / 3, "w": 1 / 3}, c, 100)]
    mean = sum(per_file) / len(per_file)

    r = run_cli(bench, corpus, "--bits")
    assert r.returncode == 0, r.stderr
    nats, bits = parse_uce(r.stdout)
    assert nats == pytest.approx(pooled, abs=5e-5)
    assert bits == pytest.approx(nats_to_bits(pooled), abs=5e-5)
    assert "[bench] 2 parquet files, 11 tokens" in r.stderr

    singles = []
    for name in ("a.parquet", "b.parquet"):
        rs = run_cli(bench / name, corpus)
        assert rs.returncode == 0, rs.stderr
        singles.append(parse_uce(rs.stdout)[0])
    assert singles == pytest.approx(per_file, abs=5e-5)
    assert abs(mean - nats) > 0.5  # the plain per-file average is a different number


# ------------------------------------------------------------
# CLI end to end
# ------------------------------------------------------------
def test_cli_end_to_end_with_and_without_bits(tmp_path):
    bench = tmp_path / "bench" / "toy" / "toy.parquet"
    write_parquet(bench, ["the cat sat", "the dog"])
    corpus = tmp_path / "corpus"
    write_parquet(corpus / "part-0.parquet", ["the cat the mat"])
    write_parquet(corpus / "sub" / "part-1.parquet", ["a mat", None, ""])

    r = run_cli(bench, corpus, "--bits")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "UCE = 1.8022 nats   (2.6000 bits)"  # same toy as the A.1 test
    assert "[bench] 1 parquet files" in r.stderr
    assert "[scan] 2 parquet files" in r.stderr
    assert "total tokens=6" in r.stderr

    r = run_cli(bench, corpus)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "UCE = 1.8022 nats"  # nats only without --bits


@pytest.mark.parametrize("case", [
    "bench_missing_column", "corpus_missing_column", "bench_unreadable",
    "corpus_unreadable", "bench_binary_column", "corpus_binary_column",
    "bench_broken_symlink", "corpus_broken_symlink", "corpus_symlink_loop",
    "corpus_zero_tokens", "missing_path", "missing_file_uri",
])
def test_bad_input_exits_nonzero_with_one_line_naming_it(tmp_path, case):
    bench, corpus = tmp_path / "bench", tmp_path / "corpus"
    write_parquet(bench / "a.parquet", ["the cat sat"])
    write_parquet(corpus / "part-0.parquet", ["the cat the mat"])
    expect = ""
    if case == "bench_missing_column":  # not the first file, so it used to be dropped silently
        bad = bench / "b.parquet"
        write_parquet(bad, ["the dog"], col="question")
    elif case == "corpus_missing_column":
        bad = corpus / "part-1.parquet"
        write_parquet(bad, ["a mat"], col="body")
    elif case == "bench_unreadable":
        bad = bench / "b.parquet"
        bad.write_bytes(NOT_PARQUET)
    elif case == "corpus_unreadable":
        bad = corpus / "part-1.parquet"
        bad.write_bytes(NOT_PARQUET)
    elif case in ("bench_binary_column", "corpus_binary_column"):  # opens fine, fails while read
        bad = (bench if case.startswith("bench") else corpus) / "z.parquet"
        pq.write_table(pa.table({"text": pa.array([b"the dog"], pa.binary())}), str(bad))
    elif case in ("bench_broken_symlink", "corpus_broken_symlink"):  # pyarrow drops it silently
        bad = (bench if case.startswith("bench") else corpus) / "z.parquet"
        os.symlink("missing.parquet", str(bad))
        expect = "broken symlink"
    elif case == "corpus_symlink_loop":  # pyarrow lists part-0 again under self/, self/self/, ...
        bad = corpus
        os.symlink(".", str(corpus / "self"))
        expect = "more than one path"
    elif case == "corpus_zero_tokens":  # used to print ln|V_b|
        bad = corpus = tmp_path / "corpus_no_words"
        write_parquet(corpus / "part-0.parquet", [None, "", "   ", "!!! ... ???"])
    elif case == "missing_path":
        bad = corpus = tmp_path / "does_not_exist"
    else:
        bad = bench = (tmp_path / "does_not_exist.parquet").as_uri()

    r = run_cli(bench, corpus, "--bits")
    assert r.returncode == 1
    assert "UCE =" not in r.stdout
    assert "Traceback" not in r.stderr
    errors = [line for line in r.stderr.splitlines() if line.startswith("[UCE] error:")]
    assert len(errors) == 1 and str(bad) in errors[0] and expect in errors[0], r.stderr


def test_directories_of_valid_symlinks_are_read(tmp_path):
    # A group directory of links to per-task files and a corpus of links to shards
    # (a symlink farm, a Hugging Face cache snapshot) read like plain files.
    blobs = tmp_path / "blobs"
    for name, texts in (("t1", ["the cat sat"]), ("t2", ["the dog"]),
                        ("s0", ["the cat the mat"]), ("s1", ["a mat"])):
        write_parquet(blobs / name, texts)
    bench, corpus = tmp_path / "bench_links", tmp_path / "corpus_links"
    for d, links in ((bench, {"t1.parquet": "t1", "t2.parquet": "t2"}),
                     (corpus, {"part-0.parquet": "s0", "sub/part-1.parquet": "s1"})):
        for name, blob in links.items():
            link_dir = os.path.dirname(str(d / name))
            os.makedirs(link_dir, exist_ok=True)
            os.symlink(os.path.relpath(str(blobs / blob), link_dir), str(d / name))

    r = run_cli(bench, corpus, "--bits")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "UCE = 1.8022 nats   (2.6000 bits)"  # same toy as the A.1 test
    assert "[bench] 2 parquet files" in r.stderr
    assert "[scan] 2 parquet files" in r.stderr


def test_file_uri_bench_path(tmp_path):
    bench = tmp_path / "bench" / "toy.parquet"
    write_parquet(bench, ["the cat sat", "the dog"])
    corpus = tmp_path / "corpus"
    write_parquet(corpus / "part-0.parquet", ["the cat the mat", "a mat"])

    r = run_cli(bench.as_uri(), corpus, "--bits")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "UCE = 1.8022 nats   (2.6000 bits)"


@pytest.mark.parametrize("n", ["0", "-1"])
def test_n_workers_below_one_is_a_usage_error(tmp_path, n):
    write_parquet(tmp_path / "bench.parquet", ["the cat sat"])
    write_parquet(tmp_path / "corpus" / "part-0.parquet", ["the cat the mat"])
    r = run_cli(tmp_path / "bench.parquet", tmp_path / "corpus", "--n_workers", n)
    assert r.returncode == 2 and "UCE =" not in r.stdout and "Traceback" not in r.stderr
    assert r.stderr.splitlines()[-1].endswith("error: --n_workers must be at least 1"), r.stderr


def test_scan_starts_at_most_one_worker_per_file(tmp_path, monkeypatch, capsys):
    import multiprocessing
    sizes = []
    real_pool = multiprocessing.Pool

    def pool(processes, *args, **kwargs):
        sizes.append(processes)
        return real_pool(processes, *args, **kwargs)

    monkeypatch.setattr(multiprocessing, "Pool", pool)
    corpus = tmp_path / "corpus"
    write_parquet(corpus / "part-0.parquet", ["the cat the mat"])
    write_parquet(corpus / "part-1.parquet", ["a mat"])
    vocab = {"the", "cat", "sat", "dog"}

    counts, T = build_train_counts_parquet(str(corpus), "text", vocab, n_workers=8)
    assert sizes == [2] and T == 6 and counts == Counter({"the": 2, "cat": 1})
    assert "[scan] 2 parquet files, 2 workers" in capsys.readouterr().err

    one_file = str(corpus / "part-1.parquet")
    counts, T = build_train_counts_parquet(one_file, "text", vocab, n_workers=8)
    assert sizes == [2, 1] and T == 2 and counts == Counter()
    assert "[scan] 1 parquet files, 1 workers" in capsys.readouterr().err


def test_hidden_and_underscore_files_skipped_on_both_sides(tmp_path):
    def make(root):
        write_parquet(root / "bench" / "a.parquet", ["the cat sat", "the dog"])
        write_parquet(root / "corpus" / "part-0.parquet", ["the cat the mat", "a mat"])
        return root / "bench", root / "corpus"

    clean_bench, clean_corpus = make(tmp_path / "clean")
    bench, corpus = make(tmp_path / "dirty")
    # Valid parquet that would change Q, c(w) and T if it were read.
    write_parquet(bench / ".hidden.parquet", ["zebra zebra"])
    write_parquet(bench / "_tmp" / "c.parquet", ["sat sat sat"])
    write_parquet(corpus / ".hidden_copy.parquet", ["dog dog dog dog"])
    write_parquet(corpus / "_temporary" / "0" / "part-1.parquet", ["sat"] * 10)
    # Files that are not parquet at all (macOS AppleDouble, Spark marker).
    (bench / "._a.parquet").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    (corpus / "._part-0.parquet").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    (corpus / "_SUCCESS").write_bytes(b"")

    assert [os.path.basename(f) for f in list_parquet_files(str(bench))] == ["a.parquet"]
    assert [os.path.basename(f) for f in list_parquet_files(str(corpus))] == ["part-0.parquet"]

    clean = run_cli(clean_bench, clean_corpus, "--bits")
    dirty = run_cli(bench, corpus, "--bits")
    assert clean.returncode == 0, clean.stderr
    assert dirty.returncode == 0, dirty.stderr
    assert dirty.stdout == clean.stdout
    assert parse_uce(dirty.stdout)[1] == pytest.approx(2.6, abs=5e-5)
    assert "[bench] 1 parquet files" in dirty.stderr
    assert "[scan] 1 parquet files" in dirty.stderr

"""Tests for benchmarks/preprocess_csr7.py.

No network: the module's `load_dataset` is replaced by a fake Hub that serves
two small examples per split, with the real column names of each dataset.
"""
import os
import re
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

from benchmarks import preprocess_csr7 as pp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "benchmarks", "preprocess_csr7.py")
CSR7 = ["arc_easy", "arc_challenge", "hellaswag", "sciq", "openbookqa", "piqa", "swag"]
OOT4 = ["lambada_openai", "commonsense_qa", "boolq", "social_iqa"]

# Per task: the Hub dataset and config the processor must request, the splits the
# Hub has, the splits the processor must read (in order), an example built from a
# tag `s` (all real columns, including those the text must leave out), and the text
# expected before sub_punct, i.e. the fields of benchmarks/README.md in order.
# FakeHub serves two examples per split, tagged with the split name and name + "2".
SPECS = {
    "arc_easy": dict(
        hub=("allenai/ai2_arc", "ARC-Easy"),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"id": "Mercury_1", "question": f"Which {s} gas do plants absorb?",
                           "choices": {"text": ["oxygen", "carbon dioxide"], "label": ["A", "B"]},
                           "answerKey": "B"},
        text=lambda s: f"Which {s} gas do plants absorb? oxygen carbon dioxide"),
    "arc_challenge": dict(
        hub=("allenai/ai2_arc", "ARC-Challenge"),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"id": "Mercury_2", "question": f"Which {s} palms warm fastest?",
                           "choices": {"text": ["dry palms", "wet palms"], "label": ["A", "B"]},
                           "answerKey": "A"},
        text=lambda s: f"Which {s} palms warm fastest? dry palms wet palms"),
    "hellaswag": dict(
        hub=("Rowan/hellaswag", None),
        splits=("train", "validation", "test"), read=("train", "validation"),
        example=lambda s: {"ind": 4, "activity_label": "Removing ice from car",
                           "ctx_a": f"The man scrapes the {s} snow.", "ctx_b": "then",
                           "ctx": f"The man scrapes the {s} snow. then",
                           "endings": [", the man adds wax.", ", the man drives off."],
                           "source_id": "activitynet~v_1", "split": s, "split_type": "indomain",
                           "label": "1"},
        text=lambda s: f"The man scrapes the {s} snow. then , the man adds wax. , the man drives off."),
    "sciq": dict(
        hub=("allenai/sciq", None),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"question": f"What {s} organism makes yogurt?", "distractor3": "viruses",
                           "distractor1": "protozoa", "distractor2": "gymnosperms",
                           "correct_answer": "mesophilic bacteria",
                           "support": "Mesophiles grow best at moderate temperature."},
        text=lambda s: (f"What {s} organism makes yogurt? mesophilic bacteria protozoa gymnosperms "
                        f"viruses Mesophiles grow best at moderate temperature.")),
    "openbookqa": dict(
        hub=("allenai/openbookqa", "main"),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"id": "7-980", "question_stem": f"The {s} sun is responsible for:",
                           "choices": {"text": ["puppies learning tricks", "plants sprouting"],
                                       "label": ["A", "B"]},
                           "answerKey": "B"},
        text=lambda s: f"The {s} sun is responsible for: puppies learning tricks plants sprouting"),
    "piqa": dict(
        hub=("ybisk/piqa", None),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"goal": f"When {s} butter is ready, you can",
                           "sol1": "Pour it onto a plate", "sol2": "Pour it into a jar", "label": 1},
        text=lambda s: f"When {s} butter is ready, you can Pour it onto a plate Pour it into a jar"),
    "swag": dict(
        hub=("allenai/swag", "regular"),
        splits=("train", "validation", "test"), read=("train", "validation", "test"),
        example=lambda s: {"video-id": "anetv_1", "fold-ind": "3416",
                           "startphrase": f"The {s} procession walks down the street. A drum line",
                           "sent1": "The procession walks down the street.", "sent2": "A drum line",
                           "gold-source": "gold", "ending0": "passes by playing.",
                           "ending1": "has heard them.", "ending2": "arrives outside.",
                           "ending3": "turns around.", "label": 0},
        text=lambda s: (f"The {s} procession walks down the street. A drum line passes by playing. "
                        f"has heard them. arrives outside. turns around.")),
    "lambada_openai": dict(
        hub=("EleutherAI/lambada_openai", "en"),
        splits=("test",), read=("test",),
        example=lambda s: {"text": f"In my palm is a clear {s} stone, and inside it is a statuette."},
        text=lambda s: f"In my palm is a clear {s} stone, and inside it is a statuette."),
    "commonsense_qa": dict(
        hub=("tau/commonsense_qa", None),
        splits=("train", "validation", "test"), read=("train", "validation"),
        example=lambda s: {"id": "075e", "question": f"The {s} sanctions seemed to what the efforts?",
                           "question_concept": "punishing",
                           "choices": {"label": ["A", "B", "C"], "text": ["ignore", "enforce", "avoid"]},
                           "answerKey": "A"},
        text=lambda s: f"The {s} sanctions seemed to what the efforts? ignore enforce avoid"),
    "boolq": dict(
        hub=("google/boolq", None),
        splits=("train", "validation"), read=("train", "validation"),
        example=lambda s: {"question": f"do iran and afghanistan speak the same {s} language",
                           "answer": True, "passage": "Persian, also known as Farsi, is a language."},
        text=lambda s: (f"Persian, also known as Farsi, is a language. "
                        f"do iran and afghanistan speak the same {s} language")),
    "social_iqa": dict(
        hub=("lighteval/siqa", None),
        splits=("train", "validation"), read=("train", "validation"),
        example=lambda s: {"context": f"Cameron had a {s} barbecue.",
                           "question": "How would Others feel as a result?",
                           "answerA": "like attending", "answerB": "like staying home",
                           "answerC": "a good friend to have", "label": "1"},
        text=lambda s: (f"Cameron had a {s} barbecue. How would Others feel as a result? "
                        f"like attending like staying home a good friend to have")),
}


class FakeHub:
    """Stands in for datasets.load_dataset and records every call.

    Serves two examples per split. Tasks in `fail` raise like an unreachable Hub;
    tasks in `empty` have all their splits but no examples; `drop` maps a task to
    the set of splits its dataset lacks. Like the real Hub, ybisk/piqa fails unless
    it is read from its parquet conversion (the main branch holds only piqa.py).
    """

    def __init__(self, fail=(), empty=(), drop=None):
        self.fail, self.empty, self.calls = set(fail), set(empty), []
        self.drop = drop or {}

    def __call__(self, path, name=None, **kwargs):
        self.calls.append((path, name, kwargs))
        task = next(t for t, spec in SPECS.items() if spec["hub"] == (path, name))
        if task in self.fail:
            raise ConnectionError(f"Couldn't reach '{path}' on the Hub")
        if path == "ybisk/piqa" and kwargs.get("revision") != "refs/convert/parquet":
            raise RuntimeError("Dataset scripts are no longer supported, but found piqa.py")
        example = SPECS[task]["example"]
        return {s: [] if task in self.empty else [example(s), example(s + "2")]
                for s in SPECS[task]["splits"] if s not in self.drop.get(task, ())}


def expected_rows(task):
    """Both examples of each split read, in split order and then example order."""
    return [pp.sub_punct(SPECS[task]["text"](tag))
            for s in SPECS[task]["read"] for tag in (s, s + "2")]


def run_main(monkeypatch, hub, *args):
    """Run preprocess_csr7.main() in-process with the fake Hub; return the exit code."""
    monkeypatch.setattr(pp, "load_dataset", hub)
    monkeypatch.setattr(sys, "argv", ["preprocess_csr7.py", *args])
    try:
        pp.main()
    except SystemExit as e:
        return e.code
    return 0


def read_text(path):
    table = pq.read_table(str(path))
    assert table.column_names == ["text"]
    return table.column("text").to_pylist()


def test_task_list_is_csr7_plus_oot4():
    assert list(pp.TASK_PROCESSORS) == CSR7 + OOT4
    assert set(SPECS) == set(pp.TASK_PROCESSORS)


def test_sub_punct_brackets_punctuation_with_newlines():
    assert pp.sub_punct("Why? (a) b, c.") == "Why\n?\n \n(\na\n)\n b\n,\n c\n.\n"
    assert pp.sub_punct("x [y] {z}; w: v!") == "x \n[\ny\n]\n \n{\nz\n}\n\n;\n w\n:\n v\n!\n"
    assert pp.sub_punct(None) == ""


@pytest.mark.parametrize("task", list(SPECS))
def test_processor_template_splits_and_hub_call(task, monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(pp, "load_dataset", hub)
    assert pp.TASK_PROCESSORS[task]() == expected_rows(task)
    # PIQA comes from the Hub's parquet conversion; nothing else passes a revision
    # or trust_remote_code.
    path, name = SPECS[task]["hub"]
    kwargs = {"revision": "refs/convert/parquet"} if task == "piqa" else {}
    assert hub.calls == [(path, name, kwargs)]


@pytest.mark.parametrize("task", list(SPECS))
def test_processor_raises_on_missing_split(task, monkeypatch):
    # The last split read is gone from the Hub dataset: the processor must not
    # return the rows of the other splits without it.
    missing = SPECS[task]["read"][-1]
    monkeypatch.setattr(pp, "load_dataset", FakeHub(drop={task: {missing}}))
    present = [s for s in SPECS[task]["splits"] if s != missing]
    with pytest.raises(ValueError, match=re.escape(
            f"split '{missing}' not found; splits present: {present}")):
        pp.TASK_PROCESSORS[task]()


def test_all_tasks_written_per_task_and_combined(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out)) == 0
    assert sorted(os.listdir(out / "benchmark_group_en")) == sorted(f"{t}.parquet" for t in SPECS)
    for task in SPECS:
        assert read_text(out / task / f"{task}.parquet") == expected_rows(task)
        assert read_text(out / "benchmark_group_en" / f"{task}.parquet") == expected_rows(task)
    captured = capsys.readouterr()
    assert "[done] outputs at" in captured.out
    assert "[warn]" not in captured.err


def test_failed_task_exits_1_other_tasks_written_no_stale_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    combined = out / "benchmark_group_en"
    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out), "--tasks", "piqa") == 0
    assert os.listdir(combined) == ["piqa.parquet"]
    capsys.readouterr()

    code = run_main(monkeypatch, FakeHub(fail={"piqa"}),
                    "--output_dir", str(out), "--tasks", "arc_easy,piqa,swag")
    assert code == 1
    err = capsys.readouterr().err
    assert "[FAIL] piqa: Couldn't reach 'ybisk/piqa' on the Hub" in err
    last = err.rstrip().splitlines()[-1]
    assert last.startswith("[error] 1 of 3 task(s) failed: ['piqa']; 2 written")
    # A rerun of piqa alone would leave only piqa in the combined directory, so the
    # last line says to rerun all three.
    assert last.endswith("; rerun with the same --tasks, since each run replaces "
                         "benchmark_group_en/")
    # The other tasks are written, and the PIQA file of the earlier run is gone
    # from the combined directory (it is not reported as another file).
    assert "[warn]" not in err
    assert sorted(os.listdir(combined)) == ["arc_easy.parquet", "swag.parquet"]
    assert read_text(out / "arc_easy" / "arc_easy.parquet") == expected_rows("arc_easy")
    assert read_text(combined / "swag.parquet") == expected_rows("swag")


def test_task_without_rows_counts_as_failed(tmp_path, monkeypatch, capsys):
    # BoolQ has both of its splits, but they hold no examples.
    out = tmp_path / "out"
    code = run_main(monkeypatch, FakeHub(empty={"boolq"}),
                    "--output_dir", str(out), "--tasks", "boolq,sciq")
    assert code == 1
    assert "[FAIL] boolq: no rows (the splits read are empty)" in capsys.readouterr().err
    assert os.listdir(out / "benchmark_group_en") == ["sciq.parquet"]
    assert not (out / "boolq").exists()


def test_missing_split_fails_the_task(tmp_path, monkeypatch, capsys):
    # BoolQ lists two splits and its Hub dataset has lost one of them: the task
    # fails instead of being written from the train split alone.
    out = tmp_path / "out"
    code = run_main(monkeypatch, FakeHub(drop={"boolq": {"validation"}}),
                    "--output_dir", str(out), "--tasks", "boolq,sciq")
    assert code == 1
    err = capsys.readouterr().err
    assert "[FAIL] boolq: split 'validation' not found; splits present: ['train']" in err
    assert err.rstrip().splitlines()[-1].startswith("[error] 1 of 2 task(s) failed: ['boolq']")
    assert os.listdir(out / "benchmark_group_en") == ["sciq.parquet"]
    assert read_text(out / "sciq" / "sciq.parquet") == expected_rows("sciq")
    assert not (out / "boolq").exists()


def test_rerun_removes_stale_task_files_from_combined_dir(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    combined = out / "benchmark_group_en"
    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out)) == 0
    (combined / "notes.txt").write_text("not written by the script")
    (combined / "extra.parquet").write_bytes(b"not written by the script")
    capsys.readouterr()

    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out), "--tasks", ",".join(CSR7)) == 0
    # Only the 7 CSR7 tasks remain; files with other names are left alone.
    assert sorted(os.listdir(combined)) == sorted([f"{t}.parquet" for t in CSR7]
                                                  + ["extra.parquet", "notes.txt"])
    log = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[combined]")]
    assert len(log) == 1 and "removed 11 file(s)" in log[0]
    assert all(f"'{t}.parquet'" in log[0] for t in OOT4)
    # Per-task folders are not touched.
    assert read_text(out / "boolq" / "boolq.parquet") == expected_rows("boolq")


def test_other_files_in_combined_dir_are_kept_with_one_warning(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    combined = out / "benchmark_group_en"
    combined.mkdir(parents=True)
    (combined / "mmlu.parquet").write_bytes(b"not written by the script")
    (combined / "notes.txt").write_text("not written by the script")
    (combined / "old").mkdir()
    (combined / "old" / "piqa.parquet").write_bytes(b"pooled too: discovery is recursive")
    (combined / ".DS_Store").write_bytes(b"skipped by --bench_path")
    (combined / "_SUCCESS").write_bytes(b"skipped by --bench_path")

    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out), "--tasks", "arc_easy") == 0
    warn = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[warn]")]
    assert warn == [f"[warn] {combined} also holds files not written by this run, and "
                    f"--bench_path pools them too: ['mmlu.parquet', 'notes.txt', 'old']"]
    # The other files are kept.
    assert sorted(os.listdir(combined)) == [".DS_Store", "_SUCCESS", "arc_easy.parquet",
                                            "mmlu.parquet", "notes.txt", "old"]
    assert (combined / "mmlu.parquet").read_bytes() == b"not written by the script"

    # With a failed task, the warning comes before the final [error] line.
    assert run_main(monkeypatch, FakeHub(fail={"boolq"}), "--output_dir", str(out),
                    "--tasks", "arc_easy,boolq") == 1
    lines = capsys.readouterr().err.rstrip().splitlines()
    assert sum(line.startswith("[warn]") for line in lines) == 1
    assert lines[-1].startswith("[error] 1 of 2 task(s) failed: ['boolq']")


def test_no_combined_neither_creates_nor_touches_combined_dir(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    combined = out / "benchmark_group_en"
    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out),
                    "--tasks", "arc_easy", "--no_combined") == 0
    assert read_text(out / "arc_easy" / "arc_easy.parquet") == expected_rows("arc_easy")
    assert not combined.exists()

    combined.mkdir()
    (combined / "boolq.parquet").write_bytes(b"earlier run")
    assert run_main(monkeypatch, FakeHub(), "--output_dir", str(out),
                    "--tasks", "arc_easy,boolq", "--no_combined") == 0
    assert os.listdir(combined) == ["boolq.parquet"]
    assert (combined / "boolq.parquet").read_bytes() == b"earlier run"

    # Without the combined directory, rerunning only the failed task is fine: the
    # last line has no rerun hint, and the directory is not checked for other files.
    capsys.readouterr()
    assert run_main(monkeypatch, FakeHub(fail={"boolq"}), "--output_dir", str(out),
                    "--tasks", "arc_easy,boolq", "--no_combined") == 1
    err = capsys.readouterr().err
    assert err.rstrip().splitlines()[-1].endswith(f"1 written under {out}")
    assert "[warn]" not in err
    assert os.listdir(combined) == ["boolq.parquet"]


def _cli(args, fail=None):
    """Run the script as a process; with `fail`, load_dataset is the fake Hub."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if fail is None:
        cmd = [sys.executable, SCRIPT] + args
    else:
        runner = (f"import runpy, sys, datasets\n"
                  f"sys.path[:0] = [{REPO!r}, {os.path.dirname(__file__)!r}]\n"
                  f"import test_benchmarks\n"
                  f"datasets.load_dataset = test_benchmarks.FakeHub(fail={fail!r})\n"
                  f"sys.argv = [{SCRIPT!r}] + {args!r}\n"
                  f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n")
        cmd = [sys.executable, "-c", runner]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


def test_cli_unknown_task_exits_2_without_writing(tmp_path):
    out = tmp_path / "out"
    proc = _cli(["--output_dir", str(out), "--tasks", "arc_easy,mmlu"])
    assert proc.returncode == 2, proc.stderr
    assert "[error] unknown tasks: ['mmlu']" in proc.stderr
    assert not out.exists()


def test_cli_failed_task_exit_status_1(tmp_path):
    out = tmp_path / "out"
    proc = _cli(["--output_dir", str(out), "--tasks", "lambada_openai,piqa"], fail=["piqa"])
    assert proc.returncode == 1, proc.stderr
    assert "[error] 1 of 2 task(s) failed: ['piqa']" in proc.stderr
    assert os.listdir(out / "benchmark_group_en") == ["lambada_openai.parquet"]

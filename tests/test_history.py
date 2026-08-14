"""Tests for the per-folder prompt history (arrow-up recall)."""

from __future__ import annotations

from agentknit_tui._history import MAX_ENTRIES, PromptHistory, history_file_for


def test_history_file_is_scoped_to_cwd(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    fa = history_file_for(tmp_path / "a")
    fb = history_file_for(tmp_path / "b")
    assert fa != fb
    assert fa.parent.name == "repl_history"
    assert fa.suffix == ".hist"

    # And resolves against the process cwd by default.
    import os

    old = os.getcwd()
    os.chdir(tmp_path / "a")
    try:
        assert history_file_for() == fa
    finally:
        os.chdir(old)


def test_record_and_recall_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    hist = PromptHistory()
    hist.record("first task")
    hist.record("second task")

    fresh = PromptHistory()  # re-read from disk
    assert len(fresh) == 2
    assert fresh[-1] == "second task"
    assert fresh[-2] == "first task"

    # The on-disk file is the same one the REPL's readline uses.
    assert fresh.path.read_text(encoding="utf-8") == "first task\nsecond task\n"


def test_record_skips_blank_and_consecutive_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hist = PromptHistory(tmp_path)
    hist.record("run tests")
    hist.record("run tests")  # consecutive duplicate dropped
    hist.record("")  # blank dropped
    hist.record("   ")  # whitespace dropped
    hist.record("ship it")
    assert len(hist) == 2
    assert hist[:] == ["run tests", "ship it"]


def test_record_multiline_prompt_splits_per_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hist = PromptHistory(tmp_path)
    hist.record("line one\nline two\n\nline three")
    assert hist[:] == ["line one", "line two", "line three"]


def test_cap_drops_oldest_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hist = PromptHistory(tmp_path)
    for i in range(MAX_ENTRIES + 25):
        hist.record(f"task {i}")
    assert len(hist) == MAX_ENTRIES
    assert hist[0] == "task 25"

    # The cap also holds after a reload from disk.
    assert len(PromptHistory(tmp_path)) == MAX_ENTRIES


def test_missing_or_corrupt_file_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hist = PromptHistory(tmp_path)
    assert len(hist) == 0  # no file yet

    hist.path.parent.mkdir(parents=True, exist_ok=True)
    hist.path.write_text("_HiStOrY_V2_\n\n  \nreal entry\n", encoding="utf-8")
    # libedit marker and blank lines skipped, entries survive.
    loaded = PromptHistory(tmp_path)
    assert loaded[:] == ["real entry"]

    # Unreadable file must not crash loading.
    hist.path.write_bytes(b"\xff\xfe garbage \x00")
    assert isinstance(PromptHistory(tmp_path)[:], list)

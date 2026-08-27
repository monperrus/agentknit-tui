"""Unit tests for the colorized unified diff shown for str_replace calls."""

from __future__ import annotations

from agentknit_tui._diff import locate_line, render_str_replace


def _spans(text: object) -> list[tuple[int, int, object]]:
    return [(s.start, s.end, s.style) for s in getattr(text, "spans", [])]


def _colors(text: object) -> set[str]:
    return {
        s.color.get_truecolor() if s.color.triplet else str(s.color.name)
        for _, _, s in _spans(text)
        if s.color is not None
    }

def test_plain_diff_lines_are_present() -> None:
    text = render_str_replace("pkg/mod.py", "a\nb\n", "a\nc\n")
    assert "--- pkg/mod.py" in text.plain
    assert "+++ pkg/mod.py" in text.plain
    assert "@@ -1,2 +1,2 @@" in text.plain
    assert "1   │ a" in text.plain       # context row keeps its marker
    assert "2 - │ b" in text.plain       # deletion
    assert "2 + │ c" in text.plain       # addition


def test_changed_lines_carry_line_colors() -> None:
    text = render_str_replace("m.py", "x = 1\n", "x = 2\n")
    colors = _colors(text)
    assert "red" in colors
    assert "green" in colors


def test_word_level_highlight_within_changed_lines() -> None:
    """Only the moved word gets the bold/background treatment."""
    text = render_str_replace("m.py", "total = count\n", "total = total_count\n")
    highlighted = [
        text.plain[s:e]
        for s, e, st in _spans(text)
        if st.bgcolor is not None
    ]
    assert highlighted, "expected word-level highlighting"
    assert any("total_count" in h or "count" in h for h in highlighted)
    # The unchanged words around it stay unhighlighted.
    joined = "".join(highlighted)
    assert "total =" not in joined


def test_word_highlight_on_both_sides() -> None:
    text = render_str_replace("m.py", "say hello to you\n", "greet you\n")
    highlighted = [text.plain[s:e] for s, e, st in _spans(text) if st.bgcolor]
    assert any("say" in h for h in highlighted)      # deleted side
    assert any("greet" in h for h in highlighted)    # added side


def test_insertion_only_is_all_green() -> None:
    text = render_str_replace("n.py", "", "fresh line\n")
    assert "1 + │ fresh line" in text.plain
    colors = _colors(text)
    assert "green" in colors
    assert "red" not in colors


def test_content_lines_starting_with_diff_markers_survive() -> None:
    """A '+'/'-' in the *content* must not be read as (or eaten by) chrome."""
    text = render_str_replace("d.py", "keep\n", "keep\n++ added\n- gone\n")
    assert "2 + │ ++ added" in text.plain
    assert "3 + │ - gone" in text.plain
    # And the mirrored deletion side.
    text = render_str_replace("d.py", "keep\n- gone\n", "keep\n")
    assert "2 - │ - gone" in text.plain


def test_line_numbers_track_the_offset() -> None:
    text = render_str_replace("d.py", "x\n", "y\n", line_offset=42)
    assert "42 - │ x" in text.plain
    assert "42 + │ y" in text.plain
    assert "@@ -42 +42 @@" in text.plain


def test_identical_strings_show_note_not_empty() -> None:
    text = render_str_replace("s.py", "same\n", "same\n")
    assert "no line-level change" in text.plain


def test_oversized_diff_is_capped() -> None:
    text = render_str_replace("big.py", "a\n" * 900, "b\n" * 900)
    lines = text.plain.splitlines()
    assert len(lines) <= 402
    assert "more lines)" in lines[-1]


def test_trailing_newline_only_change_is_noted() -> None:
    text = render_str_replace("t.py", "a\nb", "a\nb\n")
    assert "whitespace / trailing newline only" in text.plain


def test_locate_line_finds_offset(tmp_path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert locate_line(str(target), "two\n") == 2
    # Substring inside a line: counts the newline before it.
    assert locate_line(str(target), "wo\nthr") == 2
    # Missing fragment: unknown, not a bogus offset.
    assert locate_line(str(target), "nope") == 0
    # Relative path resolved against cwd.
    assert locate_line("mod.py", "three\n", cwd=str(tmp_path)) == 3
    # Unreadable file: unknown.
    assert locate_line(str(tmp_path / "gone.py"), "x") == 0


def test_locate_line_falls_back_to_new_when_edit_already_ran(tmp_path) -> None:
    """The event may be rendered after the tool ran: the file then holds
    `new`, and the line numbers must still be real, not reset to 1."""
    target = tmp_path / "mod.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert locate_line(str(target), "two\n", new="TWO\n") == 2
    # File already rewritten by the tool.
    target.write_text("one\nTWO\nthree\n", encoding="utf-8")
    assert locate_line(str(target), "two\n", new="TWO\n") == 2
    # Neither side present: unknown.
    target.write_text("other\n", encoding="utf-8")
    assert locate_line(str(target), "two\n", new="TWO\n") == 0


def test_with_file_context_pads_three_lines_each_side(tmp_path) -> None:
    from agentknit_tui._diff import with_file_context

    target = tmp_path / "mod.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 21)),
                      encoding="utf-8")
    anchored = with_file_context(str(target), "line 10\n", "line ten\n")
    assert anchored is not None
    old, new, offset = anchored
    # Three real lines before, the edit, three after.
    assert old.splitlines() == ["line 7", "line 8", "line 9", "line 10",
                                "line 11", "line 12", "line 13"]
    assert new.splitlines() == ["line 7", "line 8", "line 9", "line ten",
                                "line 11", "line 12", "line 13"]
    assert offset == 7
    text = render_str_replace(str(target), old, new, line_offset=offset)
    assert "7   │ line 7" in text.plain
    assert "10 - │ line 10" in text.plain
    assert "10 + │ line ten" in text.plain
    assert "13   │ line 13" in text.plain
    assert "line 14" not in text.plain  # context stops at three lines


def test_with_file_context_clamps_at_file_edges(tmp_path) -> None:
    from agentknit_tui._diff import with_file_context

    target = tmp_path / "mod.py"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    anchored = with_file_context(str(target), "a\n", "alpha\n")
    assert anchored is not None
    old, new, offset = anchored
    assert old.splitlines() == ["a", "b", "c"]
    assert new.splitlines() == ["alpha", "b", "c"]
    assert offset == 1


def test_with_file_context_missing_fragment_falls_back(tmp_path) -> None:
    from agentknit_tui._diff import with_file_context

    target = tmp_path / "mod.py"
    target.write_text("a\nb\n", encoding="utf-8")
    assert with_file_context(str(target), "nope\n", "x\n") is None
    assert with_file_context(str(tmp_path / "gone.py"), "a\n", "x\n") is None


def test_with_file_context_anchors_on_new_when_edit_already_ran(tmp_path) -> None:
    """Tool runs before the UI drains the event: file holds `new`, not `old`.

    The diff must still show old→new with the file's real line numbers
    and real surrounding context."""
    from agentknit_tui._diff import with_file_context

    target = tmp_path / "mod.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 21)),
                      encoding="utf-8")
    # The tool already applied the edit: line 10 is now "line ten".
    target.write_text("".join(
        f"line {'ten' if i == 10 else i}\n" for i in range(1, 21)),
        encoding="utf-8")
    anchored = with_file_context(str(target), "line 10\n", "line ten\n")
    assert anchored is not None
    old, new, offset = anchored
    assert old.splitlines()[3] == "line 10"
    assert new.splitlines()[3] == "line ten"
    assert offset == 7
    text = render_str_replace(str(target), old, new, line_offset=offset)
    assert "10 - │ line 10" in text.plain
    assert "10 + │ line ten" in text.plain
    assert "7   │ line 7" in text.plain
    assert "13   │ line 13" in text.plain

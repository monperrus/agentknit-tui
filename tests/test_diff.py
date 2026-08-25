"""Unit tests for the colorized unified diff shown for str_replace calls."""

from __future__ import annotations

from agentknit_tui._diff import render_str_replace


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
    assert "-b" in text.plain
    assert "+c" in text.plain
    assert " a" in text.plain  # context keeps its leading space


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
    assert "+fresh line" in text.plain
    colors = _colors(text)
    assert "green" in colors
    assert "red" not in colors


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

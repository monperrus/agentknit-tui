"""Tests for paste-safe copied selections.

The conversation log must never leak copy-hostile chrome: no Rich panel
borders or ``│`` gutters, and no right-hand padding — the clipboard gets
the payload text only, so pasting into an editor or shell carries no
garbage.
"""

from __future__ import annotations

from agentknit_tui.app import _clean_selection_lines


def test_panel_borders_and_gutter_are_stripped() -> None:
    lines = [
        "╭─ you ─────────────────────────────────────╮",
        "│ identify the max number of base64 chars   │",
        "│ put on an A4 paper and ocr                │",
        "╰───────────────────────────────────────────╯",
    ]
    assert _clean_selection_lines(lines) == (
        "identify the max number of base64 chars\n"
        "put on an A4 paper and ocr"
    )


def test_trailing_padding_is_stripped() -> None:
    """Rich pads wrapped lines to the block width; copies must not carry it."""
    lines = [
        "identify the max number of base64 chars   ",
        "put on an A4 paper and ocr                ",
    ]
    assert _clean_selection_lines(lines) == (
        "identify the max number of base64 chars\n"
        "put on an A4 paper and ocr"
    )


def test_partial_panel_copy_strips_gutter() -> None:
    """Selecting only the wrapped body rows still drops the ``│`` prefix."""
    lines = [
        "│ identify the max number of base64 chars   │",
        "│ put on an A4 paper and ocr                │",
    ]
    assert _clean_selection_lines(lines) == (
        "identify the max number of base64 chars\n"
        "put on an A4 paper and ocr"
    )


def test_plain_lines_pass_through_untouched() -> None:
    assert _clean_selection_lines(
        ["plain text", "  indented", "code → ok", ""]) == \
        "plain text\n  indented\ncode → ok"


def test_prompt_rules_are_dropped() -> None:
    lines = ["─" * 40, "you 14:03", "hello", "─" * 40]
    assert _clean_selection_lines(lines) == "you 14:03\nhello"


def test_user_prompt_is_dated_and_framed() -> None:
    import re

    from rich.console import Console

    from agentknit_tui.app import AgentTUI

    con = Console(width=30, record=True, file=open("/dev/null", "w"))
    con.print(AgentTUI._render_user(object.__new__(AgentTUI), "hello"))
    rows = con.export_text().splitlines()
    assert set(rows[0]) == {"─"} and set(rows[-1]) == {"─"}
    assert re.fullmatch(r"you \d\d:\d\d", rows[1])
    assert rows[2] == "hello"


def test_degenerate_inputs() -> None:
    assert _clean_selection_lines([]) == ""
    assert _clean_selection_lines(["x"]) == "x"

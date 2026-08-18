"""Tests for panel-chrome stripping on copied selections.

`SelectableRichLog.get_selection` feeds the clipboard; when the selected
lines come from a bordered Rich panel (user prompt, assistant reply, tool
output) the leading ``│`` gutter — and trailing bar plus padding — must not
leak into the copied text. Non-panel lines must pass through untouched.
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agentknit_tui.app import _strip_panel_chrome


def _panel_lines(width: int = 60) -> list[str]:
    console = Console(file=io.StringIO(), width=width, force_terminal=False)
    console.print(Panel(
        Text("identify the max number of base64 characters that can be "
             "put on an A4 paper and ocr"),
        border_style="green", title="you", title_align="left", padding=(0, 1),
    ))
    return console.file.getvalue().splitlines()  # type: ignore[attr-defined]


def test_full_panel_copy_strips_borders_and_gutter() -> None:
    out = _strip_panel_chrome(_panel_lines())
    assert out == [
        "identify the max number of base64 characters that can be",
        "put on an A4 paper and ocr",
    ]


def test_partial_panel_copy_strips_gutter() -> None:
    """Selecting only the wrapped body rows still drops the ``│`` prefix."""
    lines = _panel_lines()
    out = _strip_panel_chrome(lines[1:3])  # skip top and bottom borders
    assert out == [
        "identify the max number of base64 characters that can be",
        "put on an A4 paper and ocr",
    ]


def test_plain_lines_pass_through_untouched() -> None:
    lines = ["plain text", "  indented", "code → ok", ""]
    assert _strip_panel_chrome(lines) == lines


def test_degenerate_inputs() -> None:
    assert _strip_panel_chrome([]) == []
    assert _strip_panel_chrome(["x"]) == ["x"]
    assert _strip_panel_chrome(["│"]) == ["│"]

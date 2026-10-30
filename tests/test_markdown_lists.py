"""Regression tests for nested markdown lists in assistant messages.

An item that carries both text and a nested list used to render as a bare
``•`` with the item's own text swallowed: the renderer dropped what it
assumed was the previous child's trailing newline, but ``split_lines`` had
already consumed the terminator, so the pop ate the last segment of real
content instead.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agentknit_tui.app import _FlushMarkdown


def _render(markdown: str, width: int = 80) -> list[str]:
    buffer = StringIO()
    Console(file=buffer, width=width, force_terminal=False).print(
        _FlushMarkdown(markdown))
    return buffer.getvalue().splitlines()


def test_top_level_bullets_survive_a_nested_list() -> None:
    rows = _render(
        "- 🗑 Cleanup executed — 14 orphaned resources deleted (84d016b)\n"
        "  - 8 NSGs: frank-bump\n"
        "  - 1 NIC: javier-ron-experimen81\n"
        "- 💸 Cost picture unchanged — the cleanup saves nothing\n"
        "  - 2.34 SEK month-to-date\n"
        "- 🔒 What stays, on your instruction\n"
        "  - assert-cron-eth-diversity\n"
        "- 📉 Subscription went 37 → 23 resources\n"
    )
    assert "• 🗑 Cleanup executed — 14 orphaned resources deleted (84d016b)" in rows
    assert "• 💸 Cost picture unchanged — the cleanup saves nothing" in rows
    assert "• 🔒 What stays, on your instruction" in rows
    assert "• 📉 Subscription went 37 → 23 resources" in rows
    assert "  • 8 NSGs: frank-bump" in rows
    assert "  • 1 NIC: javier-ron-experimen81" in rows
    assert "  • 2.34 SEK month-to-date" in rows
    assert "  • assert-cron-eth-diversity" in rows


def test_wrapped_top_level_text_is_not_truncated_by_the_nested_list() -> None:
    """The pop bug ate the last wrapped line of the item's own paragraph."""
    rows = _render(
        "- 🗑 Cleanup executed — 14 orphaned resources deleted (84d016b)\n"
        "  - 8 NSGs: frank-bump\n",
        width=40,
    )
    text = "\n".join(rows)
    assert "🗑 Cleanup executed" in text
    assert "(84d016b)" in text
    assert "  • 8 NSGs: frank-bump" in rows


def test_numbered_list_with_nested_bullets() -> None:
    rows = _render(
        "1. first step\n"
        "   - detail one\n"
        "   - detail two\n"
        "2. second step\n"
    )
    assert "1. first step" in rows
    assert "   • detail one" in rows
    assert "   • detail two" in rows
    assert "2. second step" in rows

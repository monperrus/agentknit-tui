"""The TUI renders str_replace calls as a colorized unified diff, not a repr."""

from __future__ import annotations

from typing import Any

import pytest
from test_tui import _log_colors, _log_text, _make_schema, _patch_no_network

from agentknit_tui.app import SelectableRichLog


@pytest.mark.asyncio
async def test_str_replace_call_renders_unified_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    old = "def area(r):\n    return 3.14 * r * r\n"
    new = "def area(r):\n    return math.pi * r * r\n"

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_call", {
            "name": "str_replace",
            "args": {"path": "geometry.py", "old_str": old, "new_str": new},
            "fmt": "▶ str_replace(path='geometry.py', old_str='def area(r):…",
        }))
        app._event_q.put(None)
        await _wait_for(app, lambda t: "  2 + │ " in t and "math.pi" in t)

        text = _log_text(app)
        # Real diff chrome, not the engine's repr one-liner.
        assert "--- geometry.py" in text
        assert "+++ geometry.py" in text
        assert "@@" in text
        assert "  1   │ def area(r):" in text
        assert "  2 - │     return 3.14 * r * r" in text
        assert "  2 + │     return math.pi * r * r" in text
        # The repr blob from fmt is replaced entirely.
        assert "old_str=" not in text
        # Line-level red/green plus word-level highlighting.
        colors = _log_colors(app)
        assert "red" in colors
        assert "green" in colors
        log = app.query_one("#conversation", SelectableRichLog)
        bold = any(
            getattr(seg.style, "bold", False)
            for line in log.lines for seg in line
        )
        assert bold, "word-level diff highlighting expected"
        app.exit()


@pytest.mark.asyncio
async def test_other_tool_calls_still_use_engine_fmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only str_replace is rewritten; read_file keeps the generic renderer."""
    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_call", {
            "name": "read_file",
            "args": {"path": "notes.md"},
            "fmt": "▶ read_file(path='notes.md')",
        }))
        app._event_q.put(None)
        await _wait_for(app, lambda t: "read_file(path='notes.md')" in t)
        assert "--- notes.md" not in _log_text(app)
        app.exit()


async def _wait_for(app: Any, predicate: Any, *, timeout: float = 5.0) -> None:
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(_log_text(app)):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"condition not met: {predicate}")

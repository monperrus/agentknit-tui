"""The TUI renders str_replace/write_file calls as a colorized diff, not a repr."""

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
        await _wait_for(app, lambda t: "  2 +  " in t and "math.pi" in t)

        text = _log_text(app)
        # Real diff chrome, not the engine's repr one-liner.
        assert "--- geometry.py" in text
        assert "+++ geometry.py" in text
        assert "@@" in text
        assert "  1    def area(r):" in text
        assert "  2 -      return 3.14 * r * r" in text
        assert "  2 +      return math.pi * r * r" in text
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
async def test_str_replace_path_is_a_clickable_file_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diff header and title carry an OSC 8 link so terminals can open the file."""
    import os
    import socket

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
        await _wait_for(app, lambda t: "  2 +  " in t and "math.pi" in t)

        expected_uri = f"file://{socket.gethostname()}" + os.path.abspath("geometry.py")
        log = app.query_one("#conversation", SelectableRichLog)
        links = {
            link
            for line in log.lines for seg in line
            if (link := getattr(seg.style, "link", None))
        }
        assert expected_uri in links
        app.exit()


@pytest.mark.asyncio
async def test_write_file_call_renders_all_additions_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    content = "def area(r):\n    return math.pi * r * r\n"

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_call", {
            "name": "write_file",
            "args": {"path": "geometry.py", "content": content},
            "fmt": "▶ write_file(path='geometry.py', content='def area(r):…",
        }))
        app._event_q.put(None)
        await _wait_for(app, lambda t: "math.pi" in t)

        text = _log_text(app)
        assert "--- geometry.py" in text
        assert "+++ geometry.py" in text
        assert "1 +  def area(r):" in text
        # The repr blob from fmt is replaced entirely.
        assert "content=" not in text
        # write_file has no "before": every line is an addition, never a deletion.
        colors = _log_colors(app)
        assert "green" in colors
        assert "red" not in colors
        app.exit()


@pytest.mark.asyncio
async def test_write_file_path_is_a_clickable_file_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import socket

    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_call", {
            "name": "write_file",
            "args": {"path": "notes.md", "content": "hello\n"},
            "fmt": "▶ write_file(path='notes.md', content='hello\\n')",
        }))
        app._event_q.put(None)
        await _wait_for(app, lambda t: "hello" in t)

        expected_uri = f"file://{socket.gethostname()}" + os.path.abspath("notes.md")
        log = app.query_one("#conversation", SelectableRichLog)
        links = {
            link
            for line in log.lines for seg in line
            if (link := getattr(seg.style, "link", None))
        }
        assert expected_uri in links
        app.exit()


@pytest.mark.asyncio
async def test_other_tool_calls_still_use_engine_fmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only str_replace/write_file are rewritten; read_file keeps the generic renderer."""
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

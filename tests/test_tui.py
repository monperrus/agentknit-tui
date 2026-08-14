"""Offline tests for the agentknit-tui event rendering and prompt handling.

These exercise the TUI without any network or API key by stubbing
`create_client` and `_run_turn`, then driving the app via Textual's
`run_test` test harness.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

# pytest-asyncio (declared in [project.optional-dependencies].test) drives the
# async TUI tests; `asyncio_mode = "auto"` in pyproject marks every async
# test function automatically.


def _make_schema() -> dict:
    return {
        "model": "fake-1",
        "endpoint": "https://example.test/v1",
        "inferred_tool_schema": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    }


def _patch_no_network(monkeypatch: pytest.MonkeyPatch, final_text: str = "ok") -> None:
    """Replace create_client and the turn worker so tests never hit the network."""
    import agentknit_tui.app as appmod

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )
        ),
    )

    def fake_turn(self: Any, task: str) -> None:
        from agentknit_tui.app import _QueuedEvent
        self._event_q.put(_QueuedEvent("final_answer", {"text": final_text}))
        self._event_q.put(None)

    monkeypatch.setattr(appmod.AgentTUI, "_run_turn", fake_turn)


def _log_text(app: Any) -> str:
    log = app.query_one("#conversation")
    return "\n".join("".join(seg.text for seg in line) for line in log.lines)


def _log_colors(app: Any) -> set[str]:
    """Collect colour names used across the conversation log's segments."""
    colors: set[str] = set()
    log = app.query_one("#conversation")
    for line in log.lines:
        for seg in line:
            color = getattr(seg.style, "color", None)
            if color is not None and color.name:
                colors.add(color.name)
    return colors


async def _wait_for_log(
    app: Any, predicate: Any, *, timeout: float = 5.0, interval: float = 0.05
) -> None:
    """Poll the conversation log until *predicate* passes.

    Event rendering is driven by a 50 ms drain timer, so a fixed number of
    `pilot.pause()` calls races it on slow machines. Poll instead.
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(_log_text(app)):
            return
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_boot_renders_header_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        text = _log_text(app)
        assert "type a task to get started" in text
        assert "1 tools: read_file" in text
        assert "session" in text
        app.exit()


@pytest.mark.asyncio
async def test_enter_submits_and_renders_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio  # noqa: F401

    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch, final_text="echo: hi")
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        ta.load_text("hi")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        text = _log_text(app)
        assert "hi" in text          # user turn
        assert "echo: hi" in text    # assistant echo
        assert ta.text == ""         # prompt cleared after submit
        app.exit()


@pytest.mark.asyncio
async def test_mod_enter_inserts_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        # Type characters so the cursor sits at the end before the chord.
        await pilot.press("a")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert ta.text == "a\n"
        await pilot.press("b")
        await pilot.press("shift+enter")
        await pilot.pause()
        assert ta.text == "a\nb\n"
        app.exit()


@pytest.mark.asyncio
async def test_streaming_then_final_answer_no_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("content_delta", {"text": "Hello "}))
        app._event_q.put(_QueuedEvent("content_delta", {"text": "world"}))
        app._event_q.put(_QueuedEvent("content_stream_end", {}))
        app._event_q.put(_QueuedEvent("final_answer", {"text": "Hello world"}))
        app._event_q.put(None)
        await _wait_for_log(app, lambda t: t.count("Hello world") == 1)
        text = _log_text(app)
        # Streamed content renders exactly once; final_answer does not double it.
        assert text.count("Hello world") == 1
        app.exit()


@pytest.mark.asyncio
async def test_usage_goes_to_status_bar_not_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token usage must update the status bar and never print into the log."""
    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch, final_text="done")
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("usage", {
            "prompt": 1200, "completion": 300, "total": 1500,
            "cached": 800, "fmt": "\033[35m[tokens] prompt 1,200\033[0m",
        }))
        app._event_q.put(_QueuedEvent("session_usage", {
            "prompt": 1200, "completion": 300, "cached": 800,
            "fmt": "\033[35m[session tokens] prompt 1,200\033[0m",
        }))
        app._event_q.put(None)
        await _wait_for_log(app, lambda t: "1,500" in app.query_one("#status").content.plain)
        log_text = _log_text(app)
        # No token line leaks into the conversation log.
        assert "[tokens]" not in log_text
        assert "[session tokens]" not in log_text
        # The status bar reflects the totals.
        status = app.query_one("#status")
        content = status.content
        status_plain = content.plain if hasattr(content, "plain") else str(content)
        assert "1,500" in status_plain
        assert "800" in status_plain
        assert "cached" in status_plain
        app.exit()


@pytest.mark.asyncio
async def test_ctrl_c_copies_selection_instead_of_quitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl+C with an active text selection must copy, not exit the app."""
    from textual.selection import Selection

    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation")
        conv.write("Hello world copy me")
        await pilot.pause()
        # Find the rendered row and select "Hello world" within it.
        for y, strip in enumerate(conv.lines):
            row = "".join(seg.text for seg in strip)
            if "Hello world" in row:
                col = row.index("Hello")
                app.screen.selections = {
                    conv: Selection((y, col), (y, col + len("Hello world")))
                }
                break
        else:
            pytest.fail("expected row not found in conversation log")
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.clipboard == "Hello world"
        # App must still be running — the key copied instead of quitting.
        assert app.is_running
        app.exit()


@pytest.mark.asyncio
async def test_ctrl_c_without_selection_cancels_or_quits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No selection: Ctrl+C must fall through to quit (idle) as before."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert not app.screen.selections
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.return_code == 0


@pytest.mark.asyncio
async def test_paste_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal paste events land in the prompt TextArea."""
    from textual import events

    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        app.post_message(events.Paste(text="pasted text"))
        await pilot.pause()
        await pilot.pause()
        assert "pasted text" in ta.text
        app.exit()


@pytest.mark.asyncio
async def test_streamed_tool_result_renders_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamed tool results must show real output, not the placeholder.

    In the REPL, streamed tool output goes straight to stdout and the
    tool_result fmt is "(output streamed above)". The TUI never shows
    stdout, so it must re-render from the structured result payload.
    """
    import json

    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    payload = json.dumps(
        {"stdout": "file1.py\nfile2.py\n", "stderr": "", "returncode": 0},
        separators=(",", ":"),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_call", {
            "name": "shell", "args": {"command": "ls"},
            "fmt": "\033[36m▶ shell(command='ls')\033[0m",
        }))
        app._event_q.put(_QueuedEvent("tool_result", {
            "name": "shell", "result": payload, "streamed": True,
            "fmt": "\033[2m  (output streamed above)\033[0m",
        }))
        app._event_q.put(None)
        await _wait_for_log(app, lambda t: "file1.py" in t and "file2.py" in t)
        text = _log_text(app)
        assert "(output streamed above)" not in text
        assert "file1.py" in text
        assert "file2.py" in text
        # exit 0 renders green
        assert "green" in _log_colors(app)
        app.exit()


@pytest.mark.asyncio
async def test_streamed_tool_result_failure_renders_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero exit codes render red with the exit code in the title."""
    import json

    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    payload = json.dumps(
        {"stdout": "", "stderr": "boom\n", "returncode": 2},
        separators=(",", ":"),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._event_q.put(_QueuedEvent("tool_result", {
            "name": "shell", "result": payload, "streamed": True,
            "fmt": "\033[2m  (output streamed above)\033[0m",
        }))
        app._event_q.put(None)
        await _wait_for_log(app, lambda t: "boom" in t)
        text = _log_text(app)
        assert "boom" in text
        assert "exit 2" in text
        colours = _log_colors(app)
        assert "red" in colours
        assert "green" not in colours
        app.exit()


def test_build_schema_from_argv_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentknit_tui.app as appmod

    captured: dict = {}

    def fake_load(model: str, endpoint: str) -> dict:
        captured["model"] = model
        captured["endpoint"] = endpoint
        return {"model": model, "endpoint": endpoint,
                "inferred_tool_schema": []}

    monkeypatch.setattr(appmod, "load_specification", fake_load)
    schema, kwargs = appmod.build_schema_from_argv([])
    assert captured["model"] == "glm-5.2"
    assert captured["endpoint"].startswith("https://api.z.ai")
    assert schema["keyring_service"] == "z.ai"
    assert schema["keyring_username"] == "api_key"
    assert "HOME:" in kwargs["system_prompt_supplement"]
    assert kwargs["strict_cache_proof"] is True


def test_capture_slash_returns_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from agentknit.slash_commands import REGISTRY

    from agentknit_tui.app import _capture_slash

    session = {"messages": [], "model": "x", "session_id": "abc",
               "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                                "cached": 0, "cache_write": 0}}
    out = _capture_slash(REGISTRY, "/usage", session, None, "x")
    assert out is not None
    assert "Session token usage" in out
    assert "trajectory: abc" in out

    # Non-slash line is not handled.
    assert _capture_slash(REGISTRY, "hello", session, None, "x") is None
    _ = io  # silence linter about unused import intent

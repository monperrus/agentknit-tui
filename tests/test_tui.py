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


@pytest.mark.asyncio
async def test_boot_renders_header_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

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
        await pilot.pause()
        await pilot.pause()
        text = _log_text(app)
        # Streamed content renders exactly once; final_answer does not double it.
        assert text.count("Hello world") == 1
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

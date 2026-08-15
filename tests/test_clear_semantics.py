"""Offline tests for /clear and /reset-context semantics.

`/clear` and `Ctrl+L` must wipe only the displayed log; the session's
message history (the LLM context) is untouched. `/reset-context` routes to
agentknit's registry `/clear` handler, which resets the history while
keeping the system prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_tui import _log_text, _make_schema, _patch_no_network, _wait_for_log


async def _boot(monkeypatch: pytest.MonkeyPatch) -> Any:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch, final_text="unused")
    app = AgentTUI(_make_schema(), non_interactive=True)
    # Give the app a fake, non-empty history so we can observe clears.
    return app


def _messages(app: Any) -> list[dict]:
    return app._session["messages"]


async def _submit(app: Any, pilot: Any, text: str) -> None:
    ta = app.query_one("#prompt")
    ta.focus()
    await pilot.pause()
    ta.load_text(text)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_slash_clear_wipes_log_not_context(monkeypatch: pytest.MonkeyPatch) -> None:
    app = await _boot(monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        before = list(_messages(app))
        assert len(before) >= 1  # system prompt present

        # Render something into the log, then /clear it.
        app.query_one("#conversation").write("MARKER-BEFORE-CLEAR")
        await _submit(app, pilot, "/clear")
        await _wait_for_log(app, lambda t: "MARKER-BEFORE-CLEAR" not in t)

        # Log wiped, header re-rendered…
        assert "MARKER-BEFORE-CLEAR" not in _log_text(app)
        # …but the LLM context is untouched.
        assert _messages(app) == before
        app.exit()


@pytest.mark.asyncio
async def test_reset_context_clears_llm_history(monkeypatch: pytest.MonkeyPatch) -> None:
    app = await _boot(monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Seed a fake history: system + user + assistant.
        system = _messages(app)[0]
        _messages(app)[:] = [
            system,
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]

        await _submit(app, pilot, "/reset-context")
        await _wait_for_log(app, lambda t: "Context cleared" in t)

        msgs = _messages(app)
        assert [m["role"] for m in msgs] == ["system"]
        # The registry prints a confirmation, which is captured inline.
        assert "Context cleared" in _log_text(app)
        app.exit()

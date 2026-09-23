"""Prompts submitted while a turn runs become side questions.

The turn is faked as never ending (no sentinel), like one blocked in a long
tool call; ``agentknit.side_query`` is faked too — no network.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_tui import _log_text, _make_schema, _patch_no_network, _wait_for_log


def _hanging_turn(monkeypatch: pytest.MonkeyPatch, seen: list[Any]) -> None:
    import agentknit_tui.app as appmod

    def fake_turn(self: Any, task: str) -> None:
        seen.append(task)  # never posts the sentinel: the turn stays busy

    monkeypatch.setattr(appmod.AgentTUI, "_run_turn", fake_turn)


async def _submit(app: Any, pilot: Any, text: str) -> None:
    from agentknit_tui.app import AgentTUI
    ta = app.query_one("#prompt", AgentTUI.PromptInput)
    ta.focus()
    ta.load_text(text)
    await pilot.press("enter")
    await pilot.pause()


async def test_prompt_while_busy_is_answered_as_side_question(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    turns: list[Any] = []
    _hanging_turn(monkeypatch, turns)
    asked: list[tuple[Any, str]] = []

    def fake_side_query(client: Any, model: str, session: Any, question: str) -> str:
        asked.append((session, question))
        return f"still on it ({question})"

    monkeypatch.setattr(appmod, "_side_query", fake_side_query)
    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "fix the bug")
        assert app.busy and turns == ["fix the bug"]

        await _submit(app, pilot, "what are you doing?")
        await _wait_for_log(app, lambda t: "still on it (what are you doing?)" in t)
        text = _log_text(app)
        assert "you (side question)" in text
        assert "(side answer)" in text
        # Same session, no second turn, the running one is untouched.
        assert asked == [(app._session, "what are you doing?")]
        assert turns == ["fix the bug"]
        assert app.busy
        assert app.query_one("#prompt", AgentTUI.PromptInput).text == ""
        app.exit()


async def test_side_query_failure_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    _hanging_turn(monkeypatch, [])

    def boom(*_a: Any) -> str:
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(appmod, "_side_query", boom)
    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "task")
        await _submit(app, pilot, "status?")
        await _wait_for_log(app, lambda t: "Side question failed: HTTP 500" in t)
        assert app.busy
        app.exit()


@pytest.mark.parametrize("side_query_available", [True, False])
async def test_ignored_while_busy(monkeypatch: pytest.MonkeyPatch,
                                  side_query_available: bool) -> None:
    """Slash commands, or any input without agentknit.side_query, wait."""
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    _hanging_turn(monkeypatch, [])
    asked: list[str] = []
    monkeypatch.setattr(
        appmod, "_side_query",
        (lambda c, m, s, q: asked.append(q) or "x") if side_query_available else None)
    text = "/help" if side_query_available else "status?"
    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "task")
        await _submit(app, pilot, text)
        # Not consumed: the text stays in the prompt for after the turn.
        assert app.query_one("#prompt", AgentTUI.PromptInput).text == text
        assert asked == []
        app.exit()

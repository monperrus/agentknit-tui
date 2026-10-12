"""Offline tests for non-operator turns: `_begin_turn(task, sender)`.

agent-inbox's TUI mixin calls ``super()._begin_turn(task, sender)``; a
`_begin_turn` without `sender` crashed the TUI on the first inbox message.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

SENDER = {"kind": "hook", "name": "wake w-1 via inbox campaign-04"}


def _make_schema() -> dict:
    return {
        "model": "fake-1",
        "endpoint": "https://example.test/v1",
        "inferred_tool_schema": [],
    }


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentknit_tui.app as appmod

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )
        ),
    )


@pytest.mark.asyncio
async def test_subclass_passes_sender_through_begin_turn(monkeypatch) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_client(monkeypatch)
    calls: list[tuple] = []
    monkeypatch.setattr(AgentTUI, "_run_turn",
                        lambda self, *a: calls.append(a))

    class InboxTUI(AgentTUI):
        def _begin_turn(self, task: str | None,
                        sender: dict[str, str] | None = None) -> None:
            super()._begin_turn(task, sender or SENDER)

    app = InboxTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._begin_turn("it is time")
        assert calls == [("it is time", SENDER)]
        app.exit()


@pytest.mark.asyncio
async def test_operator_turn_calls_run_turn_without_sender(monkeypatch) -> None:
    """A one-argument `_run_turn` override (older subclasses, tests) keeps working."""
    from agentknit_tui.app import AgentTUI

    _patch_client(monkeypatch)
    calls: list[str | None] = []
    monkeypatch.setattr(AgentTUI, "_run_turn",
                        lambda self, task: calls.append(task))

    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._begin_turn("typed")
        assert calls == ["typed"]
        app.exit()


@pytest.mark.parametrize("takes_sender", [True, False])
@pytest.mark.asyncio
async def test_run_turn_forwards_sender_only_when_supported(
        monkeypatch, takes_sender: bool) -> None:
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_client(monkeypatch)
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(appmod.agentknit, "run_turn",
                        lambda *a, **kw: seen.append(kw))
    monkeypatch.setattr(appmod, "_RUN_TURN_TAKES_SENDER", takes_sender)

    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._begin_turn("it is time", SENDER)
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.02)
        assert len(seen) == 1
        assert ("sender" in seen[0]) is takes_sender
        if takes_sender:
            assert seen[0]["sender"] == SENDER
        app.exit()

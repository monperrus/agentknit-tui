"""Offline tests for the prompt prefill (`agentknit-tui "<task>"`)."""

from __future__ import annotations

from typing import Any

import pytest


def _make_schema() -> dict:
    return {
        "model": "fake-1",
        "endpoint": "https://example.test/v1",
        "inferred_tool_schema": [],
    }


def _patch_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

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
async def test_prefill_lands_in_prompt_not_the_model(monkeypatch) -> None:
    """Interactive mode: a CLI task prefills the prompt box; the turn starts
    only on Enter. (--non-interactive + a task is one-shot instead — see
    test_non_interactive_prefill_runs_and_exits.)"""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    submitted: list[str] = []

    def fake_run_turn(self: Any, task: str) -> None:  # noqa: ARG001
        submitted.append(task)

    monkeypatch.setattr(AgentTUI, "_run_turn", fake_run_turn)

    app = AgentTUI(_make_schema(), non_interactive=False,
                   prefill="fix the bug in foo.py")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", AgentTUI.PromptInput)
        assert prompt.text == "fix the bug in foo.py"
        assert submitted == []          # nothing sent yet
        await pilot.press("enter")
        await pilot.pause()
        assert submitted == ["fix the bug in foo.py"]
        app.exit()


@pytest.mark.asyncio
async def test_no_prefill_leaves_prompt_empty(monkeypatch) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#prompt", AgentTUI.PromptInput).text == ""
        app.exit()


@pytest.mark.asyncio
async def test_cli_argv_task_becomes_prefill(monkeypatch) -> None:
    """build_schema_from_argv: bare positionals are the task, not the model."""
    from agentknit_tui.app import build_schema_from_argv

    schema, kwargs = build_schema_from_argv(["hello", "world"])
    assert kwargs["prefill"] == "hello world"
    assert schema["model"] == "glm-5.2"

    schema, kwargs = build_schema_from_argv([])
    assert kwargs["prefill"] == ""

    schema, kwargs = build_schema_from_argv(["qwen3-8b", "https://openrouter.ai/api/v1"])
    assert kwargs["prefill"] == ""
    assert schema["model"].startswith("qwen3-8b")


@pytest.mark.asyncio
async def test_non_interactive_prefill_runs_and_exits(monkeypatch) -> None:
    """--non-interactive + a CLI task = one-shot: submit on mount, exit when
    the turn ends, and echo the final answer on stdout after run()."""
    import contextlib
    import io

    from agentknit_tui.app import AgentTUI, _QueuedEvent

    _patch_no_network(monkeypatch)

    def fake_turn(self: Any, task: str | None) -> None:
        self._event_q.put(_QueuedEvent("final_answer", {"text": "PONG"}))
        self._event_q.put(None)

    monkeypatch.setattr(AgentTUI, "_run_turn", fake_turn)

    exits: list[bool] = []
    monkeypatch.setattr(AgentTUI, "exit", lambda self, *a, **k: exits.append(True))

    app = AgentTUI(_make_schema(), non_interactive=True,
                   prefill="Reply with PONG")
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(50):  # wait for the auto-submitted turn to finish
            if not app.busy:
                break
            await pilot.pause(0.05)
        assert not app.busy
        await pilot.pause()
        assert app._last_answer == "PONG"
        # the finished one-shot turn exits the app by itself
        assert exits == [True]
        # and the answer is echoed on the plain console after teardown
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._print_one_shot_answer()
        assert "PONG" in buf.getvalue()


@pytest.mark.asyncio
async def test_interactive_prefill_waits_for_enter(monkeypatch) -> None:
    """Interactive mode keeps the review step: a prefilled task stays in the
    prompt box until the user presses Enter."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    submitted: list[str] = []

    def fake_run_turn(self: Any, task: str) -> None:  # noqa: ARG001
        submitted.append(task)

    monkeypatch.setattr(AgentTUI, "_run_turn", fake_run_turn)

    app = AgentTUI(_make_schema(), non_interactive=False,
                   prefill="fix the bug in foo.py")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert submitted == []          # nothing auto-sent
        await pilot.press("enter")
        await pilot.pause()
        assert submitted == ["fix the bug in foo.py"]
        app.exit()

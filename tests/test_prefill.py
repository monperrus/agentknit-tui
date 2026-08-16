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
    """A CLI task prefills the prompt box; the turn starts only on Enter."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    submitted: list[str] = []

    def fake_run_turn(self: Any, task: str) -> None:  # noqa: ARG001
        submitted.append(task)

    monkeypatch.setattr(AgentTUI, "_run_turn", fake_run_turn)

    app = AgentTUI(_make_schema(), non_interactive=True,
                   prefill="fix the bug in foo.py")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
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
        assert app.query_one("#prompt").text == ""
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

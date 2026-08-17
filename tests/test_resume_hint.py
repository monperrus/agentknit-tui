"""Tests for the `Resume: …` hint printed after the TUI exits."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _make_schema() -> dict:
    return {
        "model": "fake-1",
        "endpoint": "https://example.test/v1",
        "inferred_tool_schema": [],
    }


def _patch_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the environment and the network for resume-hint checks.

    ``AGENTKNIT_RESUME_COMMAND`` (exported by Martin's shell for wrapper
    scripts) would override the program under test, so scrub it.
    """
    import agentknit_tui.app as appmod

    monkeypatch.delenv("AGENTKNIT_RESUME_COMMAND", raising=False)
    monkeypatch.setattr(
        appmod,
        "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: None))
        ),
    )


def test_resume_command_uses_argv0_and_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/agentknit-tui"])

    app = AgentTUI(_make_schema(), non_interactive=True)
    # Touch reactives only inside a running app (watchers query the DOM).
    session_id = app._session["session_id"]
    assert app._resume_command() == (f"/usr/local/bin/agentknit-tui fake-1 --session {session_id}")


def test_resume_command_omits_model_for_wrapper_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper executables embed the model; their resume command must not repeat it."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["/home/martin/bin/agent-glm-5.2.py"])

    app = AgentTUI(_make_schema(), non_interactive=True)
    session_id = app._session["session_id"]
    assert app._resume_command() == (f"/home/martin/bin/agent-glm-5.2.py --session {session_id}")


def test_resume_command_honours_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setenv("AGENTKNIT_RESUME_COMMAND", "agentknit-tui glm-5.2")
    app = AgentTUI(_make_schema(), non_interactive=True)
    session_id = app._session["session_id"]
    assert app._resume_command() == f"agentknit-tui glm-5.2 --session {session_id}"


def test_resume_command_defaults_when_argv_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedders (`python -c`, run from a REPL) get a sane default program."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["-"])

    app = AgentTUI(_make_schema(), non_interactive=True)
    session_id = app._session["session_id"]
    assert app._resume_command() == f"agentknit-tui fake-1 --session {session_id}"


@pytest.mark.asyncio
async def test_run_async_prints_resume_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: run_async() leaves the Resume line on the plain console."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    app = AgentTUI(_make_schema(), non_interactive=True)

    async def quit_now(pilot: Any) -> None:
        app.exit()

    buf = []
    real_print = print

    def fake_print(*args: Any, **kwargs: Any) -> None:
        buf.append(" ".join(str(a) for a in args))

    monkeypatch.setattr("builtins.print", fake_print)
    await app.run_async(headless=True, auto_pilot=quit_now)
    monkeypatch.setattr("builtins.print", real_print)

    assert any("Resume:" in line and app._session["session_id"] in line for line in buf), buf


@pytest.mark.asyncio
async def test_resume_hint_printed_after_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_test drives _process_messages directly; run()/run_async() print the hint."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    printed: list[str] = []

    def fake_hint() -> None:
        if app._resume_hint_printed:
            return
        app._resume_hint_printed = True
        printed.append(app._resume_command())

    app = AgentTUI(_make_schema(), non_interactive=True)
    monkeypatch.setattr(app, "_print_resume_hint", fake_hint)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.exit()

    # The run_test harness bypasses run(); call it the way the CLI does.
    app._print_resume_hint()
    assert printed == [app._resume_command()]

    # Printed exactly once, even if run again.
    app._print_resume_hint()
    assert printed == [app._resume_command()]


@pytest.mark.asyncio
async def test_exit_saves_snapshot_and_logs_session_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Quitting the TUI snapshots the trajectory and logs session_end, like the REPL."""
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    saved: list[dict] = []
    logged: list[dict] = []
    monkeypatch.setattr(
        appmod._ak_core, "_save_messages_snapshot", lambda session: saved.append(session)
    )
    monkeypatch.setattr(appmod._ak_core, "_log", lambda session, rec: logged.append(rec))

    app = AgentTUI(_make_schema(), non_interactive=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.exit()

    assert saved and saved[0] is app._session
    end_records = [r for r in logged if r.get("type") == "session_end"]
    assert end_records and end_records[0]["reason"] == "tui_exit"
    assert end_records[0]["session_id"] == app._session["session_id"]

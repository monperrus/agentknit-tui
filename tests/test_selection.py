"""Tests for mouse text selection in the conversation log.

Selection must feel like a terminal (Claude Code, Codex): drag selects a
span, double click a word, triple click a line, ``ctrl+c`` copies what is
selected, and none of it takes keyboard focus away from the prompt.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from rich.text import Text
from textual import events

from agentknit_tui.app import _word_bounds


def _patch_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentknit_tui.app as appmod

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)),
        ),
    )
    monkeypatch.setattr(appmod.AgentTUI, "_run_turn", lambda self, task: None)


def _make_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    return AgentTUI(
        {"model": "fake-1", "endpoint": "https://example.test/v1",
         "inferred_tool_schema": []},
        non_interactive=True,
    )


async def _log_with(pilot: Any, app: Any, *rows: str) -> Any:
    """Empty the conversation and write one plain line per row."""
    log = app.query_one("#conversation")
    log.clear()
    for row in rows:
        log.write(Text(row))
    await pilot.pause()
    return log


@pytest.mark.parametrize(("text", "x", "expected"), [
    # A plain word.
    ("alpha beta gamma", 7, (6, 10)),
    # Clicking the first and last cell of the word is still that word.
    ("alpha beta gamma", 6, (6, 10)),
    ("alpha beta gamma", 9, (6, 10)),
    # Whitespace selects the run of whitespace.
    ("alpha beta gamma", 5, (5, 6)),
    # Paths, URLs and flags stay whole.
    ("see /usr/local/bin/x.py --flag", 10, (4, 23)),
    ("see /usr/local/bin/x.py --flag", 27, (24, 30)),
    # Past the end of the line clamps to the last word.
    ("alpha beta", 40, (6, 10)),
    # Empty line, and an empty string.
    ("", 0, (0, 0)),
    # A single character line.
    ("x", 0, (0, 1)),
])
def test_word_bounds(text: str, x: int, expected: tuple[int, int]) -> None:
    assert _word_bounds(text, x) == expected


@pytest.mark.asyncio
async def test_double_click_selects_the_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Textual's double click selects the *whole widget*, so a
    habitual double click used to replace the word with the entire
    transcript."""
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = await _log_with(pilot, app, "alpha beta gamma",
                              "second line here")
        # offset (2, 1) is the first content row (row 0 of the log).
        await pilot.click(log, offset=(2, 1), times=2)
        await pilot.pause()
        assert app.screen.get_selected_text() == "alpha"
        app.exit()


@pytest.mark.asyncio
async def test_triple_click_selects_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = await _log_with(pilot, app, "alpha beta gamma",
                              "second line here")
        await pilot.click(log, offset=(2, 1), times=3)
        await pilot.pause()
        assert app.screen.get_selected_text() == "alpha beta gamma"
        app.exit()


@pytest.mark.asyncio
async def test_clicking_the_log_keeps_prompt_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy → paste must not need a detour through a second click.

    Stock ``RichLog`` is focusable, so any click in the transcript moved
    keyboard focus there and the following paste went nowhere.
    """
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = await _log_with(pilot, app, "alpha beta gamma")
        assert app.focused is app.query_one("#prompt")
        await pilot.click(log, offset=(2, 1), times=2)
        await pilot.pause()
        assert app.focused is app.query_one("#prompt")
        app.exit()


@pytest.mark.asyncio
async def test_drag_selects_the_dragged_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = await _log_with(pilot, app, "alpha beta gamma",
                              "second line here")
        await pilot.mouse_down(log, offset=(3, 1))
        await pilot._post_mouse_events(  # noqa: SLF001
            [events.MouseMove], widget=log, offset=(12, 2))
        await pilot.mouse_up(log, offset=(12, 2))
        await pilot.pause()
        assert app.screen.get_selected_text() == "lpha beta gamma\nsecond line"
        app.exit()


@pytest.mark.asyncio
async def test_ctrl_c_copies_the_selection_instead_of_quitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch)
    copied: list[str] = []
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = await _log_with(pilot, app, "alpha beta gamma")
        log.screen.clear_selection()
        await pilot.click(log, offset=(2, 1), times=2)
        await pilot.pause()
        app._copy_selection = copied.append  # type: ignore[method-assign]
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert copied == ["alpha"]
        assert app.is_running
        app.exit()

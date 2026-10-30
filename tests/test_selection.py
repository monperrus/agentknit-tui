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


@pytest.mark.asyncio
async def test_same_line_drag_on_a_scrolled_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: on a bottom-pinned (scrolled) log a one-line drag selected
    an empty slice, so nothing could be copied.

    Textual maps a drag through the ``offset`` metadata in the rendered
    strips, which the compositor caches per scroll position; once the log
    has scrolled those offsets point at the wrong column.
    """
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = app.query_one("#conversation")
        log.clear()
        for i in range(40):
            log.write(Text(f"line {i} alpha beta gamma delta"))
        await pilot.pause()
        assert log.scroll_y > 0  # the log really is scrolled
        # Row y=6 on screen is a text row well inside the viewport.
        await pilot.mouse_down(None, offset=(4, 6))
        await pilot._post_mouse_events(  # noqa: SLF001
            [events.MouseMove], None, offset=(20, 6))
        await pilot.mouse_up(None, offset=(20, 6))
        await pilot.pause()
        selected = app.screen.get_selected_text()
        assert selected and len(selected.splitlines()) == 1
        assert "alpha" in selected
        app.exit()


@pytest.mark.parametrize(("start_x", "end_x"), [(4, 30), (30, 4)])
@pytest.mark.asyncio
async def test_drag_highlights_the_whole_span_either_way(
    monkeypatch: pytest.MonkeyPatch, start_x: int, end_x: int,
) -> None:
    """Regression: a left→right drag highlighted only its first move.

    The highlight comes from ``screen.selections``, which Textual derives
    from the ``offset`` metadata in the rendered strips; on a scrolled log
    those offsets are stale. A right→left drag came out right by accident,
    so only the direction that happened to grow from the anchor showed.
    """
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = app.query_one("#conversation")
        log.clear()
        for i in range(40):
            log.write(Text(f"line {i} alpha beta gamma delta"))
        await pilot.pause()
        await pilot.mouse_down(None, offset=(start_x, 6))
        await pilot._post_mouse_events(  # noqa: SLF001
            [events.MouseMove], None,
            offset=((start_x + end_x) // 2, 6))
        await pilot.mouse_up(None, offset=(end_x, 6))
        await pilot.pause()
        start, end = app.screen.selections[log]
        assert end.x - start.x >= abs(end_x - start_x) - 2
        assert len(app.screen.get_selected_text()) >= abs(end_x - start_x) - 2
        app.exit()


@pytest.mark.parametrize(("start_x", "end_x"), [(4, 30), (30, 4)])
@pytest.mark.asyncio
async def test_drag_highlight_follows_the_pointer(
    monkeypatch: pytest.MonkeyPatch, start_x: int, end_x: int,
) -> None:
    """Regression: mid-drag the highlight *shrank* as the pointer advanced.

    Strip offsets are cached per scroll position, so on a bottom-pinned log
    a pointer further right mapped to an earlier column and the highlighted
    span contracted until it vanished — i.e. dragging left→right never
    showed the text it was covering.
    """
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = app.query_one("#conversation")
        log.clear()
        for i in range(40):
            log.write(Text(f"line {i} alpha beta gamma delta"))
        await pilot.pause()
        step = 4 if end_x > start_x else -4
        path = [*range(start_x + step, end_x, step), end_x]
        await pilot.mouse_down(None, offset=(start_x, 6))
        await pilot.pause()
        widths: list[int] = []
        for x in path:
            await pilot._post_mouse_events(  # noqa: SLF001
                [events.MouseMove], None, offset=(x, 6))
            await pilot.pause()
            start, end = app.screen.selections[log]
            widths.append(end.x - start.x)
        # Growing all the way to the release point, never contracting.
        assert widths == sorted(widths), (path, widths)
        assert widths[-1] >= abs(end_x - start_x) - 2
        app.exit()


@pytest.mark.asyncio
async def test_drag_keeps_the_rest_of_the_line_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: everything right of the selection vanished from the strip.

    The highlight is painted by splitting the rendered line into head,
    selection and tail; ``Strip.divide`` yields one strip per cut and drops
    what follows, so the tail was empty and those cells kept a stale
    background instead of the log's own.
    """
    app = _make_app(monkeypatch)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        log = app.query_one("#conversation")
        log.clear()
        for i in range(40):
            log.write(Text(f"line {i} alpha beta gamma delta"))
        await pilot.pause()
        row = 6 - log.content_region.y
        width = log.render_line(row).cell_length  # no selection yet
        await pilot.mouse_down(None, offset=(4, 6))
        await pilot._post_mouse_events(  # noqa: SLF001
            [events.MouseMove], None, offset=(8, 6))
        await pilot.mouse_up(None, offset=(12, 6))
        await pilot.pause()

        strip = log.render_line(row)
        assert strip.cell_length == width
        selection_bg = app.screen.get_component_rich_style(
            "screen--selection").bgcolor
        unselected = "".join(
            segment.text for segment in strip._segments
            if segment.style is None or segment.style.bgcolor != selection_bg
        )
        # The part beyond the pointer is still drawn, with the normal
        # background: only the selected cells may be highlighted.
        assert "gamma delta" in unselected
        app.exit()

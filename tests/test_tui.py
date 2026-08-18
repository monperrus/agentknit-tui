"""Offline tests for the agentknit-tui event rendering and prompt handling.

These exercise the TUI without any network or API key by stubbing
`create_client` and `_run_turn`, then driving the app via Textual's
`run_test` test harness.
"""

from __future__ import annotations

import time
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
async def test_resume_session_survives_sync_event_from_init_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `--session <id>` must not crash at construction time.

    Resuming emits `session_resumed` *synchronously* from inside
    `init_session`, i.e. while `AgentTUI.__init__` is still running. The
    event queue must therefore exist before `init_session` is called.
    """

    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)

    def fake_init_session(schema: dict, *, on_event: Any = None,
                          resumed_from: str | None = None, **_: Any) -> dict:
        assert resumed_from == "4e55afe7c1ce"
        on_event("session_resumed", {
            "session_id": resumed_from,
            "messages_loaded": 3,
            "fmt": "\033[2mResumed session 4e55afe7c1ce "
                   "(3 messages in context)\033[0m",
        })
        return {"session_id": resumed_from, "model": schema["model"],
                "messages": [{"role": "user", "content": "hi"}] * 3,
                "log_path": "/tmp/does-not-matter.jsonl"}

    monkeypatch.setattr(appmod, "init_session", fake_init_session)
    app = AgentTUI(_make_schema(), non_interactive=True, session_id="4e55afe7c1ce")

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _wait_for_log(app, lambda t: "Resumed session 4e55afe7c1ce" in t)
        text = _log_text(app)
        assert "3 messages in context" in text
        assert app.session_id == "4e55afe7c1ce"
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
        from textual.geometry import Offset

        for y, strip in enumerate(conv.lines):
            row = strip.text
            if "Hello world" in row:
                col = row.index("Hello")
                app.screen.selections = {
                    conv: Selection(Offset(col, y),
                                    Offset(col + len("Hello world"), y))
                }
                break
        else:
            pytest.fail("expected row not found in conversation log")
        await pilot.pause()
        await pilot.press("ctrl+shift+c")
        await pilot.pause()
        assert app.clipboard == "Hello world"
        # App must still be running — the key copied instead of quitting.
        assert app.is_running
        app.exit()


@pytest.mark.asyncio
async def test_drag_select_yields_character_range_and_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain drag over the conversation log selects exact characters.

    The log subclass tags strips with offset metadata, so the compositor can
    map the drag to character offsets: the selection must be a real range
    (not SELECT_ALL), must highlight while dragging, and Ctrl+C must copy
    exactly the dragged span.
    """
    from textual import events
    from textual.selection import Selection

    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation")
        conv.write("Hello world copy me")
        await pilot.pause()
        # Locate the rendered row and drag across "Hello world" in it.
        # Drag coordinates are screen-space: the log's content starts at
        # content_region (border + padding shift it), and strip index wy is
        # relative to the scroll position.
        for wy, strip in enumerate(conv.lines):
            if "Hello world" in strip.text:
                col = strip.text.index("Hello")
                break
        else:
            pytest.fail("expected row not found in conversation log")
        cr = conv.content_region
        sy = cr.y + wy - conv.scroll_offset.y
        sx = cr.x + col - conv.scroll_offset.x
        await pilot.mouse_down(None, offset=(sx, sy))
        for x in range(sx + 1, sx + len("Hello world")):
            app._driver.process_message(
                events.MouseMove(widget=app.screen, x=x, y=sy,
                                 delta_x=1, delta_y=0, button=0,
                                 shift=False, meta=False, ctrl=False))
        await pilot.pause()
        await pilot.mouse_up(None, offset=(sx + len("Hello world") - 1, sy))
        await pilot.pause()
        selection = app.screen.selections.get(conv)
        assert selection is not None
        assert selection.start is not None and selection.end is not None
        assert selection != Selection(None, None)  # character range, not all
        # The dragged span must be highlighted but keep its own foreground
        # (fg == selection bg would render the text invisible).
        sel_style = next(
            s.style for s in conv.render_line(wy)._segments if s.text.strip())
        assert sel_style.bgcolor is not None
        assert sel_style.color != sel_style.bgcolor
        # Ctrl+Shift+C is the copy chord (terminals reserve plain Ctrl+C's
        # keycode differently; Shift+C is what reaches the app outside tmux).
        await pilot.press("ctrl+shift+c")
        await pilot.pause()
        assert "Hello world" in app.clipboard  # the dragged span, not the log
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


@pytest.mark.asyncio
async def test_arrow_up_recalls_prompts_from_this_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """↑ walks back through past instructions recorded for this folder."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()

        # Submit two prompts, then reload the history the prompt widget uses
        # (the app loaded it once at construction, before any submissions).
        for text in ("first task", "second task"):
            ta.load_text(text)
            await pilot.press("enter")
            await pilot.pause()
        ta.attach_history(app._history)

        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "second task"

        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "first task"

        # ↓ walks forward again, ending back at the empty prompt.
        await pilot.press("down")
        await pilot.pause()
        assert ta.text == "second task"
        await pilot.press("down")
        await pilot.pause()
        assert ta.text == ""
        app.exit()


@pytest.mark.asyncio
async def test_history_recalls_prompts_typed_in_the_repl(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Entries the line REPL wrote to its readline file are recallable in the TUI."""
    from agentknit_tui._history import history_file_for
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    hist_file = history_file_for()
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    hist_file.write_text("repl instruction\nanother one\n", encoding="utf-8")

    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "another one"
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "repl instruction"
        app.exit()


@pytest.mark.asyncio
async def test_history_parks_draft_and_survives_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """↑ parks the in-progress draft; ↓ at the end restores it."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        ta.attach_history(app._history)

        ta.load_text("draft in progress")
        await pilot.press("up")  # no history yet: ↑ must do nothing
        await pilot.pause()
        assert ta.text == "draft in progress"

        ta.load_text("first task")
        await pilot.press("enter")
        await pilot.pause()
        ta.load_text("draft in progress")
        await pilot.press("up")  # recall "first task", parking the draft
        await pilot.pause()
        assert ta.text == "first task"
        await pilot.press("down")  # forward past the oldest → draft restored
        await pilot.pause()
        assert ta.text == "draft in progress"

        # Editing a recalled entry keeps the text; ↑ starts a fresh browse.
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "first task"
        await pilot.press("space")
        await pilot.press("x")
        await pilot.pause()
        assert ta.text == "first task x"
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "first task"
        app.exit()


@pytest.mark.asyncio
async def test_arrow_up_inside_multiline_prompt_moves_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """↑ on a later line of a multiline draft must move the cursor, not recall."""
    from agentknit_tui.app import AgentTUI

    _patch_no_network(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        ta.attach_history(app._history)

        ta.load_text("first task")
        await pilot.press("enter")
        await pilot.pause()
        ta.load_text("line one\nline two")
        ta.move_cursor((1, 4))  # cursor on the second line
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "line one\nline two"  # unchanged
        assert ta.cursor_location[0] == 0  # cursor moved up a row

        # ↑ again from the first line now recalls history.
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "first task"
        app.exit()


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


@pytest.mark.asyncio
async def test_status_bar_shows_running_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While a turn runs, the line under the prompt names what it is doing."""
    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    seen: list[str] = []

    def fake_turn(self: Any, task: str) -> None:
        from agentknit_tui.app import _QueuedEvent
        seen.append(task)
        self._event_q.put(_QueuedEvent("final_answer", {"text": "done"}))
        self._event_q.put(None)

    _patch_no_network(monkeypatch)
    monkeypatch.setattr(appmod.AgentTUI, "_run_turn", fake_turn)

    app = AgentTUI(_make_schema(), non_interactive=True)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        ta = app.query_one("#prompt")
        ta.focus()
        await pilot.pause()
        ta.load_text("fix the failing login test")
        await pilot.press("enter")
        await pilot.pause()

        def status_plain() -> str:
            content = app.query_one("#status").content
            return content.plain if hasattr(content, "plain") else str(content)

        # Submitting recorded the task for the status line while busy…
        assert seen == ["fix the failing login test"]
        app.busy = True
        app._current_task = "fix the failing login test"
        app._watch_current_task()
        assert "working" in status_plain()
        assert "fix the failing login test" in status_plain()

        # The task sits on its own second line, no truncation at 120 cols.
        assert status_plain().count("\n") == 1
        assert "fix the failing login test" in status_plain().splitlines()[1]

        # A long task wraps over the full terminal width onto a second line
        # and is ellipsized at the end of it — never cut mid-way at 40 chars.
        width = app._status_width()
        assert width >= 100  # the full terminal width, not a fixed 40
        long_task = " ".join(["word"] * 100)
        app._current_task = long_task
        app._watch_current_task()
        await pilot.pause()
        width = app._status_width()  # re-read: the bar grew to two rows
        shown = status_plain()
        assert long_task not in shown
        lines = shown.splitlines()
        assert len(lines) == 3  # summary + two full-width task lines
        assert lines[0].endswith("working…")
        assert lines[1].startswith("word word")
        assert len(lines[1]) <= width
        assert len(lines[2]) <= width
        assert lines[2].endswith("…")

        # An unbreakable task fills both lines and ellipsizes at line end.
        app._current_task = "x" * 400
        app._watch_current_task()
        await pilot.pause()
        lines = status_plain().splitlines()
        assert len(lines) == 3
        assert lines[1] == "x" * width
        assert lines[2] == "x" * (width - 1) + "…"

        # A multiline task is shown as collapsed prose.
        app._current_task = "first line\nsecond line"
        app._watch_current_task()
        shown = status_plain()
        assert "first line" in shown
        assert "second line" in shown
        assert "\n" in shown

        # After the turn ends, the status line drops the task again.
        await _wait_for_log(app, lambda t: "done" in t)
        app.busy = False
        assert "first line" not in status_plain()
        assert "working" not in status_plain()
        app.exit()


def test_fit_line_truncates_at_line_end() -> None:
    from agentknit_tui.app import _fit_line

    assert _fit_line("short", 20) == "short"
    assert _fit_line("exact-10!", 10) == "exact-10!"
    assert _fit_line("0123456789A", 10) == "012345678…"
    assert _fit_line("anything", 0) == ""


"""The Textual application that hosts an agentknit session.

Design
------

The agent loop (`agentknit.run_turn`) is synchronous and blocking, and emits
typed events through a session's `on_event` callback. We:

1. Build the agentknit session up front with an `on_event` that forwards
   every event onto a thread-safe `queue.Queue`.
2. Run each turn inside a Textual worker thread (`thread=True`), so the UI
   keeps animating (spinner, keystrokes) while the model thinks.
3. Drain the event queue on a short timer in the UI thread, decoding each
   event's pre-rendered ANSI string into a Rich renderable and appending it
   to the conversation log.

This keeps the engine untouched: the TUI is a pure subscriber to the same
event stream the REPL prints.

Multi-line input uses `TextArea` so users can paste and edit freely; Enter
submits, Shift/Ctrl/Alt+Enter inserts a newline. A CancelToken is wired to a
Ctrl+C / Escape / stop binding so a turn can be interrupted cooperatively.
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import agentknit
from agentknit import (
    CancelToken,
    create_client,
    init_session,
    load_specification,
)
from agentknit import _core as _ak_core
from agentknit._core import _build_resume_cmd
from agentknit.exceptions import RateLimitError
from agentknit.slash_commands import REGISTRY as _slash_registry
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Console, Group, RenderResult
from rich.markdown import (
    BlockQuote,
    CodeBlock,
    Heading,
    ListItem,
    Markdown,
    Paragraph,
    TableElement,
)
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.geometry import Offset
from textual.message import Message
from textual.reactive import reactive
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Header, Label, RichLog, TextArea

from ._history import PromptHistory

if TYPE_CHECKING:
    from textual.events import Key

# Side questions while a turn runs need agentknit's side_query; with an older
# agentknit, input stays ignored while busy, as before.
_side_query: Callable[..., str] | None = getattr(agentknit, "side_query", None)

# ── event envelope ────────────────────────────────────────────────────────────


@dataclass
class _QueuedEvent:
    """One agentknit event, marshalled from the worker thread to the UI thread."""

    event_type: str
    data: dict


# Events whose `data["fmt"]` we print verbatim. Others are interpreted by the
# UI directly (see `_apply_event`).
_PASSTHROUGH_EVENTS = {
    "tool_call",
    "tool_result",
    "reasoning_delta",
    "content_stream_end",
    "reasoning_stream_end",
    "provider_pinned",
    "cache_cold",
    "session_resumed",
    "token_limit",
}

# Executable names that take the model as a positional argument, so the
# resume command must repeat it. Anything else (a wrapper script embedding
# the model) resumes with just `--session <id>`.
_TUI_CLI_NAMES = frozenset({"agentknit-tui"})

# Fallback width for the status bar's task line before the first layout pass.
_STATUS_FALLBACK_WIDTH = 80


# ── copy-safe markdown ────────────────────────────────────────────────────────


class _FlushMarkdown(Markdown):
    """Markdown whose rendered lines carry no copy-hostile padding.

    Rich's stock elements pad every line to the console width (paragraph
    ``justify="left"``, code blocks with ``padding=1``, headings centered),
    so a copied selection is trailed by a run of spaces on every line.
    These subclasses render flush-left/right-trimmed instead:

    * paragraphs and headings justify ``default`` (no padding),
    * code blocks render as bare syntax with zero padding,
    * block quotes keep their content but drop the ``▌`` gutter.

    Line structure (blank lines, wrapping) is identical to stock Markdown;
    only the horizontal filler differs.
    """

    class _Paragraph(Paragraph):
        def __rich_console__(self, console: Console, options: Any) -> RenderResult:
            self.text.justify = "default"
            yield self.text

    class _Heading(Heading):
        LEVEL_ALIGN = {f"h{i}": "default" for i in range(1, 7)}

    class _CodeBlock(CodeBlock):
        def __rich_console__(self, console: Console, options: Any) -> RenderResult:
            code = str(self.text).rstrip()
            yield Syntax(code, self.lexer_name, theme=self.theme,
                         word_wrap=True, padding=0)

    class _BlockQuote(BlockQuote):
        def __rich_console__(self, console: Console, options: Any) -> RenderResult:
            # Render children directly (no render_lines) so nothing is padded
            # to the console width.
            yield from console.render(self.elements, options)

    class _ListItem(ListItem):
        def render_bullet(self, console: Console, options: Any) -> RenderResult:
            # Render children without render_lines (no width padding); a
            # leading "• " marks the item, wrapped rows align under it.
            new_line = Segment("\n")
            for first, line in enumerate(
                self._child_lines(console, options, indent=2)
            ):
                yield (Segment("• ") if first == 0 else Segment("  "))
                yield from line
                yield new_line

        def render_number(self, console: Console, options: Any, number: int,
                          last_number: int) -> RenderResult:
            number_width = len(str(last_number)) + 1
            new_line = Segment("\n")
            for first, line in enumerate(
                self._child_lines(console, options, indent=number_width)
            ):
                if first == 0:
                    yield Segment(f"{number}.".rjust(number_width) + " ")
                else:
                    yield Segment(" " * (number_width + 1))
                yield from line
                yield new_line

        def _child_lines(self, console: Console, options: Any,
                         *, indent: int) -> list[list[Segment]]:
            width = max(options.max_width - indent, 1)
            out: list[list[Segment]] = []
            for child in self.elements._renderables:  # noqa: SLF001
                segments = console.render(child, options.update(width=width))
                lines = [list(line) for line in Segment.split_lines(segments)]
                if out:
                    out[-1].pop()  # drop the child's trailing newline
                out.extend(lines)
            return out

    class _TableElement(TableElement):
        def __rich_console__(self, console: Console, options: Any) -> RenderResult:
            # show_edge=False drops the blank padding rows top/bottom.
            table = Table(
                box=box.SIMPLE,
                pad_edge=False,
                style="markdown.table.border",
                show_edge=False,
                collapse_padding=True,
            )
            if self.header is not None and self.header.row is not None:
                for column in self.header.row.cells:
                    heading = column.content.copy()
                    heading.stylize("markdown.table.header")
                    table.add_column(heading)
            if self.body is not None:
                for row in self.body.rows:
                    table.add_row(*[element.content for element in row.cells])
            yield table

    elements = {**Markdown.elements,
                "paragraph_open": _Paragraph,
                "heading_open": _Heading,
                "fence": _CodeBlock,
                "code_block": _CodeBlock,
                "blockquote_open": _BlockQuote,
                "list_item_open": _ListItem,
                "table_open": _TableElement}


def _fit_line(text: str, width: int) -> str:
    """Truncate *text* to exactly *width* cells, ellipsized when it does not fit."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[:width - 1].rstrip() + "…"


# ── selectable conversation log ───────────────────────────────────────────────


class SelectableRichLog(RichLog):
    """A ``RichLog`` that supports character-accurate mouse drag selection.

    Stock ``RichLog`` renders strips without the per-cell ``offset`` style
    metadata Textual's compositor uses to map a mouse position to a
    character offset (``Log`` has it via ``Strip.apply_offsets``;
    ``RichLog`` does not). The result: any drag over the log degenerates to
    SELECT_ALL and no highlight is drawn, because the selection machinery
    cannot resolve where the selection starts or ends.

    Two overrides fix that:

    * ``render_line`` tags every strip with its (x, y) offset, so a drag
      produces a ``Selection`` with real character offsets and Textual
      highlights the selected span as you drag.
    * ``get_selection`` extracts the text from the stored strips, since the
      base ``Widget.get_selection`` inspects ``_render()``, which for a
      multi-line log is not the text either.

    Scrolling is bottom-pinned by position instead of the stock
    ``auto_scroll`` (which jumps to the end on *every* write, ripping the
    view away from whatever you scrolled up to read while the agent keeps
    streaming). :meth:`write` checks whether the viewport sits at the
    bottom *before* appending; only then does it scroll after the write.
    Scrolling up detaches (new messages append silently), scrolling back
    to the bottom re-attaches. Textual's own ``anchor()`` is not used:
    its release/restore is unreliable across wheel and key scrolls.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Pinning is driven by write() below; the stock per-write
        # scroll-to-end must stay off or it would fight the reader.
        self.auto_scroll = False

    def _pinned_to_end(self) -> bool:
        """Is the user sitting at (or scrolling toward) the bottom?"""
        if not self.size:
            return True
        # scroll_target_y, not scroll_offset: during an animated scroll the
        # target already says where the user is headed.
        return round(self.scroll_target_y) >= self.max_scroll_y

    def write(  # type: ignore[override]
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> Any:
        """Append content, following the end only if the reader is there.

        ``scroll_end=None`` (the default) applies the pin rule — follow when
        the viewport is at the bottom, hold position when it was scrolled
        up. An explicit ``True``/``False`` forces the behavior for that
        write.
        """
        follow = self._pinned_to_end() if scroll_end is None else scroll_end
        result = super().write(content, width=width, expand=expand, shrink=shrink,
                               scroll_end=False, animate=animate)
        if follow:
            # immediate=False: scroll after the refresh, when max_scroll_y
            # already accounts for the lines just appended.
            self.scroll_end(animate=animate, immediate=False, x_axis=False)
        return result

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        base = super().render_line(y).apply_offsets(scroll_x, scroll_y + y)
        selection = self.text_selection
        if selection is None:
            return base
        # While a selection is active, stylize the selected span directly
        # (offset metadata alone does not repaint with the selection style).
        span = selection.get_span(scroll_y + y)
        if span is None:
            return base
        start, end = span
        if end == -1:
            end = base.cell_length
        # Strip.divide drops cuts beyond the cell length, so the number of
        # parts varies; pad to exactly three (head, selection, tail).
        cuts = [max(0, start), max(start, end)]
        parts = list(base.divide(cuts))
        while len(parts) < 3:
            parts.append(Strip([], 0))
        head, sel, tail = parts[0], parts[1], Strip.join(parts[2:])
        # Rich style addition lets the right-hand operand win, so plain
        # apply_style(sel_style) leaves the segment's own background in
        # place (style + segment_style). Compose the other way round to
        # make the selection background visible on already-styled text —
        # but drop the selection style's *foreground*, which the theme
        # leaves as an alpha-0 color that renders as the background
        # itself (fg == bg ⇒ invisible text under selection).
        sel_style = self.screen.get_component_rich_style("screen--selection")
        bg_only = Style(bgcolor=sel_style.bgcolor, reverse=False)
        sel = Strip(
            [Segment(seg.text, seg.style + bg_only if seg.style else bg_only)
             for seg in sel._segments],
            sel.cell_length,
        )
        return Strip.join([head, sel, tail])

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        lines = self.lines
        start, end = selection
        if start is None and end is None:
            return "\n".join(strip.text for strip in lines), "\n"
        if start is None:
            start = Offset(0, 0)
        if end is None:
            end = Offset(len(lines), 1 << 30)
        if (start.y, start.x) > (end.y, end.x):
            start, end = end, start
        out: list[str] = []
        for y in range(start.y, min(end.y, len(lines) - 1) + 1):
            text = lines[y].text
            if y == start.y and y == end.y:
                text = text[start.x:end.x]
            elif y == start.y:
                text = text[start.x:]
            elif y == end.y:
                text = text[:end.x]
            out.append(text)
        return _clean_selection_lines(out), "\n"


# ── the app ───────────────────────────────────────────────────────────────────


class AgentTUI(App):
    """A Textual TUI for an agentknit coding agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #conversation {
        border: round $accent 60%;
        height: 1fr;
        padding: 0 1;
        background: $surface;
    }
    #conversation:focus {
        border: round $accent;
    }

    #prompt-row {
        height: auto;
        padding: 0 1 0 0;
    }
    #prompt-label {
        color: $accent;
        text-style: bold;
        padding: 1 0 0 0;
    }
    PromptInput {
        border: round $accent 60%;
        height: auto;
        /* Viewport-relative, not `%`: the parent #prompt-row is height:auto,
           so a percentage max-height would resolve against an unsized parent
           and clamp the input to zero rows (typing appears to do nothing). */
        max-height: 40vh;
        min-height: 3;
    }
    PromptInput:focus {
        border: round $accent;
    }

    #status {
        width: 1fr;      /* full terminal width, so the task line wraps to it */
        height: auto;
        /* No max-height + no extra padding: a taller box with bottom padding
           or a capped max-height leaves the second content row outside the
           content region, so the running-task row would render blank. */
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c,ctrl+shift+c", "cancel_or_quit", "Cancel / Quit",
                priority=True, show=False),
        Binding("escape", "maybe_cancel", "Cancel turn",
                priority=True, show=False),
        Binding("ctrl+l", "clear_log", "Clear log", show=False),
        # PageUp/PageDown always page the conversation, even while the
        # prompt holds focus — reading back during a running turn is the
        # common case, and TextArea's own page keys only move its cursor.
        Binding("pageup", "scroll_conversation('up')", "Scroll up", show=False,
                priority=True),
        Binding("pagedown", "scroll_conversation('down')", "Scroll down",
                show=False, priority=True),
    ]

    title = "agentknit"
    sub_title = "agent"

    # Reactive state mirrors of the session, for the status bar.
    model: reactive[str] = reactive("…", layout=False)
    session_id: reactive[str] = reactive("…", layout=False)
    busy: reactive[bool] = reactive(False, layout=False)
    prompt_tokens: reactive[int] = reactive(0, layout=False)
    completion_tokens: reactive[int] = reactive(0, layout=False)
    cached_tokens: reactive[int] = reactive(0, layout=False)

    def __init__(
        self,
        schema: dict,
        *,
        non_interactive: bool = False,
        session_id: str | None = None,
        system_prompt_supplement: str = "",
        max_output_tokens: int | None = None,
        strict_cache_proof: bool = True,
        min_cacheable_tokens: int | None = None,
        prefill: str = "",
    ) -> None:
        super().__init__()
        self._schema = schema
        self._non_interactive = non_interactive
        self._session_id_in = session_id
        self._supplement = system_prompt_supplement
        self._max_output_tokens = max_output_tokens
        self._strict_cache_proof = strict_cache_proof
        self._prefill = prefill

        # Event plumbing — must exist *before* init_session: resuming a
        # session (--session <id>) emits `session_resumed` synchronously from
        # inside init_session, and _on_event enqueues onto _event_q.
        self._event_q: queue.Queue[_QueuedEvent | None] = queue.Queue()
        self._decoder = AnsiDecoder()
        self._cancel_token: CancelToken | None = None

        # The prompt of the turn currently running, shown in the status bar
        # so the line under the input reminds what the agent is working on.
        self._current_task: str = ""

        # Streaming buffers: deltas arrive one chunk at a time without newlines,
        # so we accumulate until the matching *_stream_end event flushes a line.
        self._content_buf: list[str] = []
        self._reasoning_buf: list[str] = []
        # Tracks whether content_delta was seen for the *current* turn, so
        # final_answer can skip re-printing text that was already streamed.
        self._streamed_content: bool = False

        # Last 10% band of the token budget seen (0..10); the `[budget]`
        # countdown is echoed to the log only when usage crosses into a new
        # decile, not after every LLM call.
        self._last_budget_decile: int | None = None

        self._client = create_client(schema)
        self._history = PromptHistory()
        self._resume_hint_printed = False
        self._ran_interactive = False
        self._session = init_session(
            schema,
            non_interactive=non_interactive,
            resumed_from=session_id,
            system_prompt_supplement=system_prompt_supplement,
            max_output_tokens=max_output_tokens,
            strict_cache_proof=strict_cache_proof,
            min_cacheable_tokens=min_cacheable_tokens,
            on_event=self._on_event,
        )
        self._model_name = self._session.get("model") or schema.get("model", "agent")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical():
            yield SelectableRichLog(id="conversation", highlight=False, markup=False,
                                    wrap=True,
                                    classes="conversation-log")
            with Vertical(id="prompt-row"):
                yield self.PromptInput(
                    "", id="prompt", classes="PromptInput", soft_wrap=True,
                    placeholder="task… (Enter submit · "
                                "Shift/Ctrl/Alt+Enter newline)",
                )
            yield self.StatusBar(id="status")
        # 50 ms timer to drain queued events from the worker thread.
        self.set_interval(0.05, self._drain_events)

    def on_mount(self) -> None:
        display_name = self._schema.get("display_name") or f"agentknit {self._model_name}"
        self.title = display_name
        self.sub_title = self._model_name
        self.model = self._model_name
        self.session_id = self._session.get("session_id", "…")

        log = self.query_one("#conversation", SelectableRichLog)
        log.write(self._header_block())

        tool_names = [
            ((t.get("function") or t).get("name", "?"))
            for t in (self._session.get("tools") or self._schema.get("inferred_tool_schema") or [])
        ]
        if tool_names:
            log.write(Text(f"{len(tool_names)} tools: {', '.join(tool_names)}",
                           style="dim"))
        log.write(Text(
            f"session {self.session_id} · log {_session_log_path(self._session)}",
            style="dim"))
        log.write(Text(
            "Enter submits · Shift/Ctrl/Alt+Enter newline · ↑/↓ history "
            "· Esc cancels · Ctrl+Shift+C copies selection (or cancels) · /help · /exit",
            style="dim italic"))
        log.write(Text(""))

        self.query_one("#prompt", AgentTUI.PromptInput).attach_history(self._history)
        prompt = self.query_one("#prompt", TextArea)
        if self._prefill:
            # One-shot task passed on the CLI: land it in the prompt for the
            # user to review, not straight into the model.
            prompt.load_text(self._prefill)
            prompt.cursor_location = (0, len(self._prefill))
        prompt.focus()

    # ── agentknit event subscription (runs on the worker thread) ──────────────

    def _on_event(self, event_type: str, data: dict) -> None:
        """Thread-safe sink: enqueue every event for the UI thread to drain."""
        self._event_q.put(_QueuedEvent(event_type, data))

    @work(thread=True, exclusive=True, group="turn")
    def _run_turn(self, task: str | None) -> None:
        """Run one agentknit turn on a background thread."""
        self._cancel_token = CancelToken()
        # Pause the global signal-based SIGINT handler while a turn runs in
        # a thread: it kills subprocesses directly and would race with the
        # textual event loop. We rely on the CancelToken instead.
        prev_sigint = _steal_sigint()
        try:
            try:
                agentknit.run_turn(
                    self._client, self._model_name, self._session, task,
                    cancel=self._cancel_token,
                )
            except KeyboardInterrupt:
                # Cancellation or a real interrupt — surface a notice.
                self._event_q.put(_QueuedEvent("interrupted", {}))
            except RateLimitError as exc:
                # No retry-after info was given — the engine already stopped
                # the loop instead of guessing a delay. Surface it plainly.
                # HTTP 403 carries its own wording ("Quota exhausted …"),
                # which providers like Kimi use for an exhausted window.
                label = "Quota" if getattr(exc, "status_code", 429) == 403 else "Rate limited"
                text = f"{label}: {exc}"
                self._event_q.put(_QueuedEvent(
                    "error", {"text": text, "fmt": f"\033[31m{text}\033[0m"}))
            except Exception as exc:  # noqa: BLE001 — engine surfaces its own errors too
                self._event_q.put(_QueuedEvent(
                    "error", {"text": str(exc),
                              "fmt": f"\033[31mError: {exc}\033[0m"}))
            finally:
                with contextlib.suppress(Exception):
                    _ak_core._save_messages_snapshot(self._session)
        finally:
            _restore_sigint(prev_sigint)
            self._cancel_token = None
            self._event_q.put(None)  # sentinel: turn finished

    @work(thread=True, group="side")
    def _run_side_query(self, question: str) -> None:
        """Answer a side question on its own thread, next to the running turn.

        ``agentknit.side_query`` reads a snapshot of the session and never
        writes to it, so the turn thread is unaffected.  The answer is
        written straight to the log (not via the event queue, whose sentinel
        belongs to the turn).
        """
        assert _side_query is not None
        log = self.query_one("#conversation", SelectableRichLog)
        try:
            answer = _side_query(self._client, self._model_name, self._session,
                                 question)
            rendered: Any = self._render_assistant(answer or "(empty answer)",
                                                   side=True)
        except Exception as exc:  # noqa: BLE001 — shown, the turn goes on
            rendered = Text(f"Side question failed: {exc}", style="red")
        self.call_from_thread(log.write, rendered)

    # ── UI-thread drain loop ──────────────────────────────────────────────────

    def _drain_events(self) -> None:
        """Render queued events until the turn sentinel is seen."""
        if self._event_q.empty():
            return
        log = self.query_one("#conversation", SelectableRichLog)
        saw_sentinel = False
        while True:
            try:
                item = self._event_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                saw_sentinel = True
                continue
            self._apply_event(item, log)

        if saw_sentinel:
            self._end_turn_ui()

    def _apply_event(self, ev: _QueuedEvent, log: RichLog) -> None:
        et, data = ev.event_type, ev.data
        fmt = data.get("fmt")

        # Mirror usage totals into the status bar, then suppress the
        # per-message `[tokens]` line — usage lives in the status bar only.
        if et in ("usage", "session_usage"):
            for k, attr in (("prompt", "prompt_tokens"),
                            ("completion", "completion_tokens"),
                            ("cached", "cached_tokens")):
                if k in data:
                    setattr(self, attr, int(data[k] or 0))
            return

        # The `[budget]` countdown fires on every LLM call; echo it to the
        # log only when usage crosses into a new decile of the budget (i.e.
        # another 10% of the window was consumed since the last line), so
        # it stays informative instead of repeating after each message.
        if et == "token_budget":
            if self._budget_decile_crossed(data):
                log.write(self._ansi(fmt) if fmt else Text(
                    f"[budget] {data.get('remaining', 0):,}/"
                    f"{data.get('budget', 0):,} tokens remaining",
                    style="magenta"))
            return

        if et == "content_delta":
            self._streamed_content = True
            self._content_buf.append(data.get("text", ""))
            return
        if et == "content_stream_end":
            text = "".join(self._content_buf).strip()
            self._content_buf = []
            if text:
                log.write(self._render_assistant(text))
            return

        if et == "reasoning_delta":
            self._reasoning_buf.append(data.get("text", ""))
            return
        if et == "reasoning_stream_end":
            text = "".join(self._reasoning_buf).rstrip()
            self._reasoning_buf = []
            if text:
                log.write(Text(text, style="dim italic"))
            return

        if et == "final_answer":
            text = (data.get("text") or "").strip()
            # Streaming sessions already flushed the answer via content_delta;
            # avoid printing it twice.
            if text and not self._content_was_streamed_this_turn():
                log.write(self._render_assistant(text))
            return

        if et == "interrupted":
            log.write(Text("[interrupted]", style="yellow"))
            return

        if et == "rate_limit_wait":
            # agentknit emits this before sleeping after a retryable HTTP 429.
            # It is a state transition, not an error: keep the running turn
            # active, but make the otherwise silent wait visible in the log.
            if fmt:
                log.write(self._ansi(fmt))
            else:
                delay = data.get("delay")
                resume_at = data.get("resume_at")
                message = "Rate limited — waiting for the provider limit to reset"
                if delay is not None:
                    message += f" ({float(delay):.1f}s)"
                if resume_at:
                    message += f"; resuming at {resume_at}"
                log.write(Text(message, style="yellow"))
            return

        if et == "error":
            if fmt:
                log.write(self._ansi(fmt))
            else:
                log.write(Text(data.get("text", "error"), style="red"))
            return

        if et in _PASSTHROUGH_EVENTS and fmt:
            # The engine's `fmt` replaces streamed output with the placeholder
            # "(output streamed above)" — the live lines went to stdout, which
            # the TUI never shows. Render the full result from the event data
            # instead so tool output is visible in the conversation log.
            if et == "tool_result" and data.get("streamed"):
                log.write(self._render_tool_result(data))
                return
            if et == "tool_call" and data.get("name") == "str_replace":
                # `repr()` of old_str/new_str is unreadable; show a real diff.
                log.write(self._render_str_replace_call(data))
                return
            log.write(self._ansi(fmt))
            return

        # Anything else with a fmt string: render it. Events without fmt are
        # pure-data notifications the UI consumes structurally elsewhere.
        if fmt:
            log.write(self._ansi(fmt))

    def _content_was_streamed_this_turn(self) -> bool:
        return self._streamed_content

    def _budget_decile_crossed(self, data: dict) -> bool:
        """True when usage crossed into a new decile of the token budget.

        The engine emits ``token_budget`` after every LLM call; printing it
        each time drowns the conversation. We show the countdown once per
        10% band of the budget, re-arming when usage drops back (after a
        compaction or a context reset).
        """
        try:
            used = int(data.get("used", 0) or 0)
            budget = int(data.get("budget", 0) or 0)
        except (TypeError, ValueError):
            return False
        if budget <= 0:
            return False
        decile = min(10, used * 10 // budget)
        crossed = (self._last_budget_decile is not None
                   and decile != self._last_budget_decile)
        self._last_budget_decile = decile
        return crossed

    def _end_turn_ui(self) -> None:
        self.busy = False
        self._current_task = ""
        self._streamed_content = False
        self._watch_current_task()
        self.query_one("#prompt", TextArea).focus()
        self.refresh()

    # ── input handling ────────────────────────────────────────────────────────

    class PromptSubmitted(Message):
        """Posted by the prompt widget when the user submits input."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class PromptInput(TextArea):
        """A TextArea tuned for prompt entry.

        ``Enter`` submits (posts :class:`AgentTUI.PromptSubmitted`) rather than
        inserting a newline; newline insertion is moved to the Mod+Enter
        chords (``Shift``/``Ctrl``/``Alt``+``Enter``). Without this override
        ``TextArea._on_key`` would consume bare ``Enter`` to insert a ``\\n``
        before our app-level handler runs.

        History recall borrows readline's ergonomics: ``up`` on the first
        line walks back through previously submitted prompts (one per press,
        even if an entry wraps across several display rows), ``down`` walks
        forward again, and any other key returns to live editing while
        keeping the recalled text in place.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._hist: PromptHistory | None = None
            self._hist_index: int | None = None  # None: not browsing
            self._hist_draft: str = ""  # in-progress text saved on first ↑

        def attach_history(self, history: PromptHistory) -> None:
            self._hist = history

        async def _on_key(self, event: Key) -> None:  # type: ignore[override]
            key = event.key
            if key in ("shift+enter", "ctrl+enter", "alt+enter"):
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return
            if key == "enter":
                event.prevent_default()
                event.stop()
                self.reset_history_browsing()
                self.post_message(AgentTUI.PromptSubmitted(self.text))
                return
            if key == "up":
                first_row = self.selection.start[0] == 0
                if first_row and self._history_prev():
                    event.prevent_default()
                    event.stop()
                    return
            elif key == "down":
                last_row = self.selection.start[0] == self.document.line_count - 1
                if last_row and self._history_next():
                    event.prevent_default()
                    event.stop()
                    return
            # Any edit leaves history browsing but keeps the recalled text in
            # place — pressing ↑ again re-enters browsing from the top.
            if self._hist_index is not None and key not in ("up", "down"):
                self._hist_index = None
            # Defer everything else (printable chars, arrows, backspace, …)
            # to TextArea's default key handling.
            await super()._on_key(event)

        # ── history navigation ────────────────────────────────────────────────

        def _history_prev(self) -> bool:
            """Step one entry back in history; False if already oldest."""
            if not self._hist:
                return False
            newest = len(self._hist) - 1
            if self._hist_index is None:
                if newest < 0:
                    return False
                self._hist_draft = self.text  # entering browsing: park the draft
                self._hist_index = newest
            elif self._hist_index > 0:
                self._hist_index -= 1
            else:
                return False
            self._load_history_entry(self._hist[self._hist_index])
            return True

        def _history_next(self) -> bool:
            """Step one entry forward; False when back to live editing."""
            if not self._hist or self._hist_index is None:
                return False
            if self._hist_index >= len(self._hist) - 1:
                # Past the oldest-known entry: back to the parked draft.
                self._hist_index = None
                self._load_history_entry(self._hist_draft)
                return True
            self._hist_index += 1
            self._load_history_entry(self._hist[self._hist_index])
            return True

        def reset_history_browsing(self) -> None:
            """Drop back to live editing (called after a submit)."""
            self._hist_index = None
            self._hist_draft = ""

        def _load_history_entry(self, text: str) -> None:
            """Replace the prompt contents with *text*, cursor to the end."""
            self.load_text(text)
            self.move_cursor(self.document.end, center=True)

    @on(TextArea.Changed)
    def _maybe_clear_idle_hint(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt":
            return
        # no-op: reserved for future input-length mirroring in the status bar.
        return

    def on_key(self, event: Key) -> None:
        # Key handling for the prompt lives on PromptInput itself (it must
        # override TextArea._on_key, which consumes bare Enter before app-level
        # handlers run). Nothing to do here; kept for future global shortcuts.
        return

    def _handle_prompt_submitted(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        # While a turn runs (e.g. blocked in a long tool call), plain text is
        # a side question answered from the conversation so far; slash
        # commands still wait for the turn to end.
        side = self.busy
        if side and (text.startswith("/") or _side_query is None):
            return

        # Reset the prompt for the next turn.
        prompt = self.query_one("#prompt", AgentTUI.PromptInput)
        prompt.reset_history_browsing()
        prompt.load_text("")

        log = self.query_one("#conversation", SelectableRichLog)
        log.write(self._render_user(text, side=side))

        if side:
            self._history.record(text)
            self._run_side_query(text)
            return

        # Persist for arrow-up recall in this folder (shared with the REPL).
        self._history.record(text)

        lowered = text.lower()
        if lowered in ("/exit", "/quit", "exit", "quit"):
            self.exit()
            return
        if lowered == "/clear":
            # The TUI intercepts /clear: wipe the displayed log only. The
            # agent's message history (LLM context) is untouched, but the
            # status bar counters go with the visible log — users read them
            # as "tokens of what's on screen".
            self._reset_usage_status()
            log.clear()
            # The pin rule cannot see "bottom" on an empty log the same way
            # a fresh boot does; force the jump so the banner is in view.
            log.write(self._header_block(), scroll_end=True)
            return
        if lowered == "/reset-context":
            # Route to the registry's real /clear handler, which resets the
            # session message history (keeping the system prompt) and the
            # usage/compaction counters — then refresh the log too.
            text = "/clear"

        # Slash command? agentknit's registry prints to stdout; capture it.
        if text.startswith("/"):
            retried = False

            def _on_continue() -> None:
                nonlocal retried
                retried = True

            out = _capture_slash(
                _slash_registry, text, self._session, self._client, self._model_name,
                on_continue=_on_continue,
            )
            if out:
                log.write(self._ansi(out) if "\033[" in out else Text(out, style="cyan"))
            if not retried:
                return
            # `/c` prints nothing — it just asks for the interrupted turn to
            # be retried with no new user message appended.
            self._begin_turn(None)
            return

        # Real turn.
        self._begin_turn(text)

    def _begin_turn(self, task: str | None) -> None:
        """Set busy state and kick off `task` (or a `/c` retry) as a turn.

        `_current_task` first: watch_busy refreshes the status bar
        synchronously and must already see the running task.
        """
        self._current_task = task or "(retrying interrupted turn)"
        self.busy = True
        self._content_buf = []
        self._reasoning_buf = []
        self._streamed_content = False
        self._run_turn(task)

    @on(PromptSubmitted)
    def _on_prompt_submitted(self, message: PromptSubmitted) -> None:
        self._handle_prompt_submitted(message.text)

    # ── actions (bindings) ────────────────────────────────────────────────────

    def _copy_selection(self, text: str) -> None:
        """Copy ``text`` to the system clipboard.

        ``App.copy_to_clipboard`` only writes an OSC 52 escape and hopes the
        terminal stack honors it — VTE-based terminals (Terminator, older
        gnome-terminal) silently ignore it, so the copy vanishes. Fall back
        to the platform clipboard tool when one is available (xclip/xsel,
        wl-copy, pbcopy, clip.exe); if none is, keep the OSC 52 route.
        """
        self.copy_to_clipboard(text)
        for tool, args in (
            ("xclip", ("xclip", "-selection", "clipboard", "-in")),
            ("xsel", ("xsel", "--clipboard", "--input")),
            ("wl-copy", ("wl-copy",)),
            ("pbcopy", ("pbcopy",)),
            ("clip.exe", ("clip.exe",)),
        ):
            if shutil.which(tool):
                try:
                    subprocess.run(args, input=text.encode("utf-8"),
                                   check=True, timeout=2,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                except (OSError, subprocess.SubprocessError):
                    continue  # try the next tool / keep OSC 52 only
                return

    def action_cancel_or_quit(self) -> None:
        # If the user has a text selection active — in the conversation log
        # *or in the focused prompt* — the copy chord should copy it rather
        # than quit/cancel. Textual's built-in copy lives on a non-priority
        # binding that our priority binding shadows, so we re-route here.
        #
        # The prompt comes first: a TextArea tracks its selection in its own
        # `selection` reactive and never registers it in screen.selections
        # (it clears screen selections on every cursor move), so
        # `_selected_text` cannot see it.
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self._copy_selection(focused.selected_text)
            return
        selected = self._selected_text()
        if selected:
            self._copy_selection(selected)
            self.screen.clear_selection()
            return
        if self.busy:
            self.action_maybe_cancel()
        else:
            self.exit()

    def on_unmount(self) -> None:
        """Teardown mirrored from the REPL's `_repl_teardown`.

        Saves the resume snapshot and logs ``session_end``; the resume hint
        itself is printed later, after Textual restored the plain console.
        """
        with contextlib.suppress(Exception):
            _ak_core._save_messages_snapshot(self._session)
        with contextlib.suppress(Exception):
            _ak_core._log(self._session, {"type": "session_end",
                                          "session_id": self._session.get("session_id"),
                                          "reason": "tui_exit"})

    def _resume_command(self) -> str:
        """The command that resumes this session, like the REPL's hint.

        Reuses agentknit's ``_build_resume_cmd`` so the
        ``AGENTKNIT_RESUME_COMMAND`` override keeps working. The model
        argument is included only when we were launched as the
        ``agentknit-tui`` entry point (which takes one); wrapper scripts
        embed the model already.
        """
        program = sys.argv[0] if sys.argv else ""
        if not program or program in ("-", ""):
            program = "agentknit-tui"  # e.g. run via `python -c` / embedders
        include_model = Path(program).name in _TUI_CLI_NAMES
        session_id = self._session.get("session_id") or ""
        return _build_resume_cmd(self._model_name, session_id,
                                 default_program=program,
                                 include_model=include_model)

    def _print_resume_hint(self) -> None:
        """Echo the resume command on the plain console, like the REPL.

        Called after :meth:`run`/:meth:`run_async` return — Textual has
        already torn its driver down, so the line lands in the shell, right
        below where the TUI was. Only printed once per process.
        """
        if self._resume_hint_printed:
            return
        self._resume_hint_printed = True
        from agentknit._core import DIM, RESET

        print(f"\n{DIM}Resume: {self._resume_command()}{RESET}", flush=True)

    def run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            return super().run(*args, **kwargs)
        finally:
            self._print_resume_hint()

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            return await super().run_async(*args, **kwargs)
        finally:
            self._print_resume_hint()

    def action_maybe_cancel(self) -> None:
        if self._cancel_token is not None and not self._cancel_token.cancelled:
            self._cancel_token.cancel()
            self.query_one("#conversation", SelectableRichLog).write(
                Text("[cancelling…]", style="yellow"))

    def action_clear_log(self) -> None:
        self._reset_usage_status()
        log = self.query_one("#conversation", SelectableRichLog)
        log.clear()
        log.write(self._header_block(), scroll_end=True)

    def action_scroll_conversation(self, direction: str) -> None:
        """Page the conversation log without leaving the prompt.

        Works mid-turn: paging up detaches from the end so streaming
        output no longer drags the view back down; paging down re-attaches
        once it reaches the bottom.
        """
        log = self._conversation_log()
        page = log.scrollable_content_region.height
        if direction == "up":
            log.scroll_page_up(animate=False)
        else:
            # Page down from the current target; if less than a full page
            # remains, snap to the bottom so the pin rule re-attaches (a
            # plain page would stop short — content keeps growing while the
            # reader is detached).
            y = round(log.scroll_target_y) + page
            if log.max_scroll_y - y < page:
                y = log.max_scroll_y
            log.scroll_to(y=y, animate=False)

    def _reset_usage_status(self) -> None:
        """Zero the token counters shown in the status bar.

        Called when the visible log is wiped (``/clear``, ``Ctrl+L``): the
        counters are read as tokens of the conversation on screen. The
        session's own ``usage_totals`` (what ``/usage`` and the trajectory
        journal report) are untouched; the next ``usage`` event re-syncs the
        status bar from the session.
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0

    # ── status bar / watchers ─────────────────────────────────────────────────

    def watch_busy(self, busy: bool) -> None:  # noqa: D401
        self.sub_title = (f"{self._model_name} · working…"
                          if busy else self._model_name)
        self._refresh_status()
        self.refresh()

    def _watch_current_task(self) -> None:
        """`_current_task` is plain state (not reactive); refresh when set."""
        self._refresh_status()

    # ── status bar ────────────────────────────────────────────────────────────

    class StatusBar(Label):
        """The status bar under the prompt: model, working directory, tokens, task.

        Two lines when a turn is running: the summary row, then the running
        task wrapped across the full terminal width. One line when idle.

        Subscribed to the app's reactives via :meth:`AgentTUI._refresh_status`;
        token usage is fed here from the ``usage`` / ``session_usage`` events
        instead of being printed into the conversation log.
        """

    def _conversation_log(self) -> RichLog:
        return self.query_one("#conversation", SelectableRichLog)

    def _selected_text(self) -> str | None:
        """Text currently selected in the conversation log, if any.

        Delegates to the log's own ``get_selection`` (see
        :class:`SelectableRichLog`), which extracts the span from the stored
        strips. Any widget in the screen may hold a selection (e.g. the
        prompt's own TextArea), so consult all of them, but only copy from
        selectable-text widgets — a TextArea copies itself via its own
        Ctrl+C binding before this fallback runs.
        """
        screen = self.screen
        if not screen.selections:
            return None
        chunks: list[str] = []
        for widget, selection in screen.selections.items():
            if not widget.is_attached:
                continue
            text = widget.get_selection(selection)
            if text is None:
                continue
            extracted, _sep = text
            if extracted.strip():
                chunks.append(extracted.rstrip("\n"))
        return "\n".join(chunks) or None

    def _status_text(self) -> Text:
        parts = [Text(self.model, style="bold")]
        parts.append(Text(" - ", style="dim"))
        parts.append(Text(os.path.basename(os.getcwd()) or os.getcwd(), style="dim"))
        total = self.prompt_tokens + self.completion_tokens
        parts.append(Text(" - ", style="dim"))
        parts.append(Text(f"{total:,} tokens", style="cyan"))
        if self.cached_tokens:
            pct = self.cached_tokens / total * 100 if total else 0
            parts.append(Text(f" ({self.cached_tokens:,} cached, {pct:.1f}%)", style="cyan"))
        if self.busy:
            parts.append(Text(" - ", style="dim"))
            parts.append(Text("working…", style="yellow"))
        status = Text.assemble(*parts)
        if self.busy and self._current_task:
            status.append("\n")
            status.append(Text(self._task_lines().plain, style="italic"))
        return status

    # ── status bar task line ──────────────────────────────────────────────────

    def _status_width(self) -> int:
        """Cells available to the status bar's task line, minus its padding."""
        try:
            bar = self.query_one("#status", Label)
        except Exception:
            return _STATUS_FALLBACK_WIDTH
        # `content_region` excludes the horizontal padding (0 1), so this is
        # the real per-line width of the terminal.
        return bar.content_region.width or _STATUS_FALLBACK_WIDTH

    def _task_lines(self) -> Text:
        """The running task, word-wrapped over at most two terminal lines."""
        width = self._status_width()
        if width <= 0:
            return Text("")

        # Collapse newlines: the status bar shows a prompt as running prose.
        task = " ".join(self._current_task.split())
        if not task:
            return Text("")

        from rich.console import Console

        lines = list(Text(task).wrap(Console(), width, overflow="fold"))
        if len(lines) > 2:
            # Keep two lines and mark the cut with an ellipsis at the line
            # end: append when there is room, replace the last cell when the
            # wrap already filled the line.
            second = lines[1].plain.rstrip() + " …"
            lines = lines[:2]
            lines[1] = Text(second if len(second) <= width
                            else second[:width - 1] + "…")
        return Text("\n".join(line.plain for line in lines))

    def on_resize(self) -> None:
        # The task line wraps to the terminal width; re-wrap on resizes.
        if self.busy and self._current_task:
            self._refresh_status()

    def _refresh_status(self) -> None:
        self.query_one("#status", Label).update(self._status_text())

    def watch_model(self, value: str) -> None:
        self._refresh_status()

    def watch_session_id(self, value: str) -> None:
        self._refresh_status()

    def watch_prompt_tokens(self, value: int) -> None:
        self._refresh_status()

    def watch_completion_tokens(self, value: int) -> None:
        self._refresh_status()

    def watch_cached_tokens(self, value: int) -> None:
        self._refresh_status()

    # ── rendering helpers ─────────────────────────────────────────────────────

    def _ansi(self, s: str) -> Any:
        """Decode an ANSI-escaped string into a Rich renderable.

        agentknit's `fmt` strings are raw `\033[…` escapes (no Rich markup),
        so AnsiDecoder is the faithful path.
        """
        segments = list(self._decoder.decode(s))
        if len(segments) == 1:
            return segments[0]
        return Group(*segments)

    def _render_str_replace_call(self, data: dict) -> Any:
        """Render a ``str_replace`` tool call as a colorized unified diff.

        The engine's generic formatter prints `str_replace(path=…,
        old_str='…', new_str='…')` with repr()-quoted bodies, which is
        unreadable for any real edit. The structured args let us rebuild
        the diff: red deletions, green additions, and word-level
        highlighting so the exact changed words pop out.
        """
        args = data.get("args") or {}
        path = str(args.get("path") or "?")
        old = args.get("old_str")
        new = args.get("new_str")
        if not isinstance(old, str) or not isinstance(new, str):
            return self._ansi(data.get("fmt", ""))

        from ._diff import locate_line, render_str_replace, with_file_context

        anchored = with_file_context(path, old, new, cwd=os.getcwd())
        if anchored is not None:
            old, new, line_offset = anchored
        else:
            line_offset = locate_line(path, old, new=new, cwd=os.getcwd()) or 1
        body = render_str_replace(path, old, new, line_offset=line_offset)
        return Group(
            Text(f"⟨str_replace {path}⟩", style="bold magenta"),
            body,
        )

    def _render_tool_result(self, data: dict) -> Any:
        """Render the full body of a streamed tool result.

        For streamed results the engine's ``fmt`` is just the placeholder
        ``(output streamed above)`` because the real lines went straight to
        stdout in the REPL. The TUI never shows stdout, so re-render from the
        event's structured ``result`` payload (JSON for shell tools, plain
        text otherwise), capped like the engine's own formatter.

        Green when the command exited 0, red otherwise.
        """
        import json

        name = data.get("name", "tool")
        raw = data.get("result") or ""
        if not isinstance(raw, str):
            # Tool/adapter bugs can surface non-str results (booleans,
            # dicts); render them instead of crashing on splitlines().
            raw = str(raw)

        body = raw
        returncode = 0
        # Shell tool results are JSON envelopes; unwrap for readability.
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                chunks: list[str] = []
                if parsed.get("stdout"):
                    chunks.append(parsed["stdout"].rstrip())
                if parsed.get("stderr"):
                    chunks.append(parsed["stderr"].rstrip())
                if isinstance(parsed.get("matches"), list):
                    chunks.extend(m.get("text", "") for m in parsed["matches"])
                if chunks:
                    body = "\n".join(chunks)
                rc = parsed.get("returncode")
                if isinstance(rc, int):
                    returncode = rc
        except (ValueError, TypeError):
            pass

        lines = body.splitlines()
        if len(lines) > 40:
            body = "\n".join(lines[:40]) + f"\n… ({len(lines) - 40} more lines)"

        ok = returncode == 0
        colour = "green" if ok else "red"
        title = f"⟨{name} output⟩" if body else f"⟨{name}: no output⟩"
        if not ok:
            # Escape the bracket: "[exit 2]" would parse as Rich markup.
            title += f" \\[exit {returncode}]"

        return Text.assemble(
            (title + "\n", f"bold {colour}"),
            (body, f"dim {colour}"),
        )

    def _header_block(self) -> Text:
        return Text.assemble(
            (self.title + "\n", "bold blue"),
            ("type a task to get started\n", "dim blue"),
        )

    def _render_user(self, text: str, *, side: bool = False) -> Text:
        label = "you (side question)\n" if side else "you\n"
        return Text.assemble((label, "bold green"), (text, ""))

    def _render_assistant(self, text: str, *, side: bool = False) -> Any:
        # Render assistant prose as Markdown, flush-trimmed so copies carry
        # no padding (see _FlushMarkdown). Falls back to plain Text if the
        # MD parse fails.
        try:
            # code_theme=ansi_*: transparent background — the syntax renderer
            # pads lines to full width otherwise (copy-hostile padding), and
            # ANSI colors keep the TUI's own palette.
            body: Any = _FlushMarkdown(text, code_theme="ansi_dark")
        except Exception:  # noqa: BLE001
            body = Text(text)
        label = f"{self._model_name} (side answer)" if side else self._model_name
        return Group(Text(label, style="bold cyan"), body)


# ── helpers ───────────────────────────────────────────────────────────────────


def _session_log_path(session: dict) -> str:
    try:
        return str(session.get("log_path") or "—")
    except Exception:  # noqa: BLE001
        return "—"


# Vertical border characters of Rich's box-drawing variants, plus the
# single-cell left/right corner glyphs — any of these glued to the payload
# means "panel chrome leaked into the copy".
_PANEL_VBORDERS = "│┃║"


def _clean_selection_lines(lines: list[str]) -> str:
    """Make copied selection lines paste-safe.

    Rich pads every rendered line of a wrapped block to the block's width,
    and Textual's ``RichLog.write`` pads each stored strip again to the
    render width — so a raw copy trails every line with spaces (garbage on
    paste, especially into editors that highlight trailing whitespace).
    Legacy panels (now retired) could additionally leak their ``│`` gutter;
    both are stripped here so the clipboard gets the payload only:

    * every line is right-trimmed of padding,
    * a leading panel gutter (``│`` + one padding space) is dropped,
    * full border rows (``╭─…─╮`` with any title) are removed entirely.
    """
    out: list[str] = []
    for ln in lines:
        stripped = ln.rstrip()
        if stripped.lstrip()[:1] in "╭╰┌└╶╰":
            continue  # border row (top/bottom edge, with any title)
        if stripped[:1] in _PANEL_VBORDERS:
            stripped = stripped[1:]
            if stripped[:1] == " ":
                stripped = stripped[1:]
        stripped = stripped.rstrip(" │┃║╮╯┐┘").rstrip()
        out.append(stripped)
    return "\n".join(out).rstrip("\n")


def _capture_slash(registry: Any, line: str, session: dict, client: Any,
                   model: str, *,
                   on_continue: Callable[[], None] | None = None) -> str | None:
    """Run a slash command, capturing its printed output.

    agentknit's slash handlers `print(...)` their result, so we redirect
    stdout around the dispatch and return the captured text. Returns None if
    the line wasn't a slash command. *on_continue* is forwarded to the
    registry — see :meth:`SlashCommandRegistry.dispatch`.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handled = registry.dispatch(line, session, client, model, on_continue=on_continue)
    if not handled:
        return None
    return buf.getvalue().rstrip()


# ── SIGINT management ─────────────────────────────────────────────────────────


def _steal_sigint() -> Any:
    """Temporarily reset SIGINT to default while a turn runs in a thread.

    agentknit installs a module-level handler that kills the active tool
    subprocess on SIGINT. That handler is meant for the foreground REPL;
    inside Textual's worker thread we drive cancellation via CancelToken
    instead, so we neutralise it for the turn's duration.
    """
    import signal

    try:
        return signal.signal(signal.SIGINT, signal.SIG_DFL)
    except (ValueError, OSError):
        # Not in the main thread (Textual workers) — nothing to reset.
        return None


def _restore_sigint(prev: Any) -> None:
    import signal

    if prev is None:
        return
    try:
        signal.signal(signal.SIGINT, prev)
    except (ValueError, OSError):
        pass


# ── convenience CLI plumbing exported for the entry point ─────────────────────


def build_schema_from_argv(argv: list[str]) -> tuple[dict, dict]:
    """Parse the subset of flags shared with agent-glm style wrappers.

    Returns (schema, kwargs) where kwargs holds the agentknit `run_task` /
    `init_session` knobs the TUI honours. Unknown flags are ignored so this
    stays compatible with arbitrary wrapper scripts.
    """
    flags_with_value = {"--session", "--max-tokens", "--system-prompt-supplement",
                        "--cache-key"}
    non_interactive = "--non-interactive" in argv
    strict = "--no-strict-cache-proof" not in argv
    session_id: str | None = None
    max_tokens: int | None = None
    supplement = ""
    cache_key: str | None = None
    positional: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--session":
            session_id = _next(argv, arg)
            skip = True
            continue
        if arg == "--max-tokens":
            max_tokens = _to_int(_next(argv, arg))
            skip = True
            continue
        if arg == "--system-prompt-supplement":
            supplement = _next(argv, arg) or ""
            skip = True
            continue
        if arg == "--cache-key":
            cache_key = _next(argv, arg)
            skip = True
            continue
        if arg.startswith("--"):
            if arg in flags_with_value:
                skip = True
            continue
        positional.append(arg)
    _ = flags_with_value  # documented for readers

    # Default: try to load a glm-5.2 spec so `agentknit-tui` works with no args,
    # matching the agent-glm-5.2.py wrapper that motivated this TUI.
    model = "glm-5.2"
    endpoint = "https://api.z.ai/api/coding/paas/v4"
    prefill = ""
    if positional:
        model, endpoint_arg = _split_model_endpoint(positional[0])
        if endpoint_arg:
            endpoint = endpoint_arg
        elif len(positional) >= 2 and _looks_like_endpoint(positional[1]):
            # Two-arg form: `agentknit-tui <model> <endpoint>`
            endpoint = positional[1]
        else:
            # Not a "model endpoint" pair: the positionals are the task to
            # prefill (agent-glm wrapper semantics), model stays the default.
            model = "glm-5.2"
            prefill = " ".join(positional)
    schema = load_specification(model, endpoint)

    # Inject keyring config for z.ai when using the default spec, mirroring
    # agent-glm-5.2.py. Harmless for other providers.
    if endpoint.startswith("https://api.z.ai"):
        schema.setdefault("keyring_service", "z.ai")
        schema.setdefault("keyring_username", "api_key")

    _home = os.path.expanduser("~")
    _cwd = os.getcwd()
    supplement = (supplement + "\n\n" if supplement else "") + (
        f"Environment paths (do NOT assume or guess these — use the values below):\n"
        f"- HOME: {_home}\n"
        f"- Current working directory: {_cwd}\n"
        f"Never assume the home directory is /home/user; it is {_home}."
    )

    kwargs = dict(
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=supplement,
        max_output_tokens=max_tokens,
        strict_cache_proof=strict,
        cache_key=cache_key,
        prefill=prefill,
    )
    return schema, kwargs


def _next(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (IndexError, ValueError):
        return ""


def _to_int(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _split_model_endpoint(arg: str) -> tuple[str, str]:
    parts = shlex.split(arg)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return arg, ""


def _looks_like_endpoint(arg: str) -> bool:
    """True when *arg* is clearly an URL, not a task word."""
    return arg.startswith(("http://", "https://", "run://"))

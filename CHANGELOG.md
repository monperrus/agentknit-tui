# Changelog

## Unreleased

- An exhausted provider quota reported as HTTP 403 (Kimi's Coding Plan
  endpoint does this instead of 429) is now labelled `Quota: …` and shows
  the provider's own message; agentknit waits through the window when the
  provider says when it reopens, instead of ending the session on an
  opaque `403 Client Error: Forbidden`.

- Reading no longer fights the agent: scrolling the conversation up
  (mouse wheel or `PageUp`) detaches the view from the end, so new
  messages append silently instead of yanking you back to the bottom —
  you can keep reading while the turn keeps streaming. Scrolling back to
  the bottom (or `PageDown` within a page of it) re-attaches the
  auto-follow. This replaces Textual's `auto_scroll`, which jumped to
  the end on every write.
- `PageUp`/`PageDown` now page the conversation even while the prompt
  holds focus (they previously only moved the prompt's own cursor).

- Paste-safe rendering: the conversation log no longer draws Rich panels
  around user prompts, assistant replies, tool output or diffs — the `│`
  gutters and border rows copied as garbage. Blocks are now delimited by a
  single styled heading line (`you`, the model name, `⟨tool output⟩`,
  `⟨str_replace path⟩`).
- Paste-safe copies everywhere: every copied selection line is right-trimmed
  of the padding Rich pads blocks to (the clipboard gets the payload only),
  and the `str_replace` diff gutter no longer uses a `│` separator between
  the line number/marker and the content.
- Assistant markdown renders flush: paragraphs, headings, lists, quotes,
  tables and code blocks no longer pad each line to the terminal width, so
  copying a reply yields the text without trailing-space filler.
- The `[budget]` token countdown is now echoed to the conversation log only
  when usage crosses into a new decile of the context budget (i.e. after
  another 10% was consumed), instead of after every LLM call.
- Fixed the running task no longer appearing in the status bar while a turn
  runs: the bar's `max-height: 2` combined with the busy-state bottom
  padding pushed the task row out of the content region, so it painted
  blank even though the label held the text. The bar now grows naturally
  (summary row + task row, no cap, no extra padding).
- Fixed `str_replace` diff line numbers resetting to 1 when the tool had
  already rewritten the file by the time the event was rendered (the UI
  drains events after the tool executes): the diff now anchors on the
  replacement text in that case, so the line-number gutter and context
  lines stay real either way.
- `str_replace` diffs now show up to three lines of real context before
  and after each change, read from the target file — like `diff -u`
  instead of an isolated fragment. Falls back to the bare fragment when
  the file is unreadable or no longer contains the old text.
- `str_replace` tool calls now render as a colorized unified diff (red
  deletions, green additions) with word-level highlighting inside each
  changed line pair, replacing the engine's `repr()` one-liner.
- The diff gained a line-number gutter showing real file line numbers
  (located by reading the target file; relative paths also tried against
  the working directory). Rows are built from `SequenceMatcher` opcodes
  instead of parsed diff text, so content lines that themselves start
  with `+`/`-` no longer masquerade as diff chrome or get swallowed by
  it — the root cause of the doubled-marker `++` rows.

## 0.1.0

First public release of the Textual TUI front-end for
[agentknit](https://pypi.org/project/agentknit/) coding agents.

- Persistent conversation pane (selectable, scrollable) replacing the
  line-based REPL: user turns, assistant answers, streamed reasoning and
  streamed tool output rendered inline.
- Multiline prompt: Enter submits, Shift/Ctrl/Alt+Enter insert a newline,
  arrow-up/down recalls instructions previously given in the same folder.
- Live status bar: model, session id, token usage (incl. cached tokens) and,
  while a turn runs, the task wrapped over two terminal-width lines.
- Character-accurate mouse drag selection with Ctrl+C / Ctrl+Shift+C copy
  via the platform clipboard (xclip, xsel, wl-copy, pbcopy, clip.exe) with
  an OSC 52 fallback.
- Cooperative cancellation of a running turn (Escape / Ctrl+C) through
  agentknit's `CancelToken`; rate-limit errors surfaced distinctly.
- Session persistence and resume (`--session <id>`), with the resume command
  printed to the console on exit, plus `AGENTKNIT_RESUME_COMMAND` override
  for wrapper scripts.
- Slash commands (`/help`, `/usage`, `/clear`, `/reset-context`, `/compact`,
  `/model`, `/exit`) forwarded to agentknit's registry, output captured
  inline; Ctrl+L clears the displayed log.
- One-shot task prefill from the CLI (`agentknit-tui <model> "task…"`).

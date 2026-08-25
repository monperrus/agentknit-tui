# Changelog

## Unreleased

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

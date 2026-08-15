# agentknit-tui

A [Textual](https://textual.textualize.io/) TUI front-end for
[agentknit](https://github.com/monperrus/agentknit) coding agents.

Features: 
- one always-visible conversation pane
- a multiline prompt at the bottom, 
- tool calls / results / token accounting rendered
inline
- a live status bar. 

the TUI is a pure subscriber to the event stream from agentknit.

## Install

```
pip install agentknit-tui
```

It depends on `agentknit`, `textual`, and `rich`.

## Usage

```
agentknit-tui                       # default: glm-5.2 via z.ai
agentknit-tui glm-5.2
agentknit-tui "qwen3-8b" "https://openrouter.ai/api/v1"
agentknit-tui --session <id>        # resume a previous trajectory
agentknit-tui --non-interactive     # drop ask_user* tools from the schema
agentknit-tui --no-strict-cache-proof
```

### Key bindings

| key                  | action                          |
| -------------------- | ------------------------------- |
| `Enter`              | submit the prompt               |
| `Shift/Ctrl/Alt+Enter` | insert a newline              |
| `↑` / `↓`            | recall past prompts (same folder, shared with the REPL) |
| `Esc` / `Ctrl+C`     | cancel the running turn (or quit when idle) |
| `Ctrl+Shift+C`       | copy the mouse selection in the log          |
| `Ctrl+L`             | clear the on-screen conversation log only |

### Mouse

- **Drag** over the conversation log selects text in place (highlighted as
  you drag); `Ctrl+Shift+C` then copies the selection (plain `Ctrl+C` also
  works when your terminal delivers it). `Esc` clears the selection.
- Paste into the prompt with the terminal's own paste (`Ctrl+Shift+V`,
  `Shift+Insert`, or middle-click) — `Ctrl+V` inside the TUI only reads
  text copied *within* the TUI.
- Copying tries the platform clipboard tool first (`xclip`/`xsel`, `wl-copy`,
  `pbcopy`, `clip.exe`) and also emits OSC 52; one of the two reaches the
  system clipboard even in VTE terminals that ignore OSC 52 (Terminator,
  older gnome-terminal).

`Ctrl+L` (and the TUI's built-in `/clear` alias) wipes the displayed log;
the agent's message history — the context sent to the model — is untouched.
To reset the LLM context (session history, keeping the system prompt), run
`/reset-context` in the TUI; it forwards to agentknit's `/clear` handler.

Slash commands (`/help`, `/usage`, `/clear`, `/compact`, `/model`, `/reset-context`, `/exit`)
are forwarded to agentknit's command registry; their printed output is
captured and shown inline.

On exit the TUI prints the command to resume the session on the console,
exactly like the end of the agentknit REPL:

```
Resume: agentknit-tui glm-5.2 --session <session-id>
```

Set `AGENTKNIT_RESUME_COMMAND` to override the program prefix (wrappers
embedding the model already do this); the `--session <id>` suffix is always
appended.

## Design notes

- The agent loop (`agentknit.run_turn`) is synchronous and blocking and
  emits typed events through a session `on_event` callback. The TUI runs
  each turn in a Textual worker thread and forwards every event onto a
  thread-safe queue; a 50 ms UI-thread timer drains the queue and renders
  each event's pre-formatted ANSI string into a Rich renderable.
- agentknit's `fmt` strings are raw `\033[…]` ANSI escapes (not Rich
  markup), so they are decoded with `rich.ansi.AnsiDecoder` for faithful
  colour/style reproduction.
- Multi-line input uses `TextArea`, so users can paste and edit freely.
  A `CancelToken` is wired to the cancel bindings so a turn can be
  interrupted cooperatively, exactly like Ctrl+C in the REPL.
- Prompt history is per-folder and shared with the REPL: both read and
  write agentknit's readline history file
  (`~/.local/share/agent_probe/repl_history/<md5(cwd)>.hist`), so arrow-up
  recalls instructions typed in either front-end, scoped to the folder they
  were typed in. `↑` on the first prompt line walks back, `↓` forward, and
  any edit drops back to normal typing.
- On exit the TUI mirrors the REPL's teardown: the trajectory snapshot is
  saved (if the session has any messages), a `session_end` record is
  appended to the session log, and — after Textual restores the plain
  console — the same dim `Resume: …` line the REPL prints is echoed so the
  session can be continued with `--session <id>`.
- agentknit installs a module-level SIGINT handler (meant for the
  foreground REPL) that kills the active tool subprocess. The TUI
  neutralises it for each turn's duration and drives cancellation through
  the `CancelToken` instead.

## License

MIT.

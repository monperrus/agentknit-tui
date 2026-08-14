# agentknit-tui

A [Textual](https://textual.textualize.io/) TUI front-end for
[agentknit](https://github.com/monperrus/agentknit) coding agents.

It replaces the classic line-based REPL (`agentknit.run_repl`) with a
persistent terminal UI: one always-visible conversation pane, a multiline
prompt at the bottom, tool calls / results / token accounting rendered
inline, and a live status bar. The agent loop itself is unchanged — the TUI
is a pure subscriber to the same `on_event` stream the REPL already prints.

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
| `Esc` / `Ctrl+C`     | cancel the running turn (or quit when idle) |
| `Ctrl+L`             | clear the conversation log      |

Slash commands (`/help`, `/usage`, `/clear`, `/compact`, `/model`, `/exit`)
are forwarded to agentknit's command registry; their printed output is
captured and shown inline.

### Embedding

Wrapper scripts that already build their own schema can use the app directly:

```python
from agentknit import load_specification
from agentknit_tui import AgentTUI

schema = load_specification("glm-5.2", "https://api.z.ai/api/coding/paas/v4")
schema["keyring_service"]  = "z.ai"
schema["keyring_username"] = "api_key"

AgentTUI(schema, non_interactive=True).run()
```

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
- agentknit installs a module-level SIGINT handler (meant for the
  foreground REPL) that kills the active tool subprocess. The TUI
  neutralises it for each turn's duration and drives cancellation through
  the `CancelToken` instead.

## License

MIT.

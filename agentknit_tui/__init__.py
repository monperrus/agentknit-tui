"""agentknit-tui — a Textual TUI front-end for agentknit coding agents.

This package wraps `agentknit`'s session/turn engine and renders its event
stream inside a persistent terminal UI (a transcribed log + a multiline
prompt), replacing the classic line-based REPL exposed by `run_repl`.

The whole point is: one always-visible conversation pane, a rich prompt at
the bottom, tool calls / results / token accounting inline, and a live
status bar. The agent loop itself is unchanged — the TUI is a pure
subscriber to the same `on_event` stream the REPL already uses.
"""

from __future__ import annotations

from .app import AgentTUI

__version__ = "0.1.0"
__all__ = ["AgentTUI", "__version__"]

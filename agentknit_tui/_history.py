"""Per-folder prompt history, shared with agentknit's line REPL.

The REPL persists its readline history under
``~/.local/share/agent_probe/repl_history/<md5(cwd)[:12]>.hist`` — one entry
per line of the file. The TUI reads and writes that very same file, so arrow-up
recalls instructions typed in *either* front-end, scoped to the folder they
were typed in.

Format notes
------------

GNU readline stores each entry as one raw line (embedded newlines are *not*
escaped, so a multiline entry round-trips as separate entries); libedit — the
readline Python module on macOS — merely prefixes its files with a
``_HiStOrY_V2_`` marker, which we skip. Entries are capped at the same 500
the REPL configures via ``readline.set_history_length(500)``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["MAX_ENTRIES", "PromptHistory", "history_file_for"]

MAX_ENTRIES = 500  # mirrors the REPL's readline.set_history_length(500)

# libedit (macOS) writes this marker as the first line of its history files.
_LIBEDIT_HEADER = "_HiStOrY_V2_"


def history_file_for(cwd: str | os.PathLike[str] | None = None) -> Path:
    """Return the history file path for *cwd* (default: the process cwd)."""
    cwd = os.fspath(cwd if cwd is not None else os.getcwd())
    tag = hashlib.md5(cwd.encode()).hexdigest()[:12]
    return (Path.home() / ".local" / "share" / "agent_probe"
            / "repl_history" / f"{tag}.hist")


class PromptHistory:
    """The prompt history for the current working directory.

    Loaded once at startup and appended to on every submission; the file is
    rewritten whole so the on-disk cap matches the in-memory one.
    """

    def __init__(self, cwd: str | os.PathLike[str] | None = None) -> None:
        self.path = history_file_for(cwd)
        self._entries = self._load()

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> str:
        return self._entries[index]

    # ── loading / saving ──────────────────────────────────────────────────────

    def _load(self) -> list[str]:
        """Read entries (oldest first) from disk; missing/corrupt file → []."""
        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        entries = [line for line in raw.splitlines()
                   if line.strip() and line != _LIBEDIT_HEADER]
        return entries[-MAX_ENTRIES:]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text("".join(e + "\n" for e in self._entries),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            # History is a convenience — never fail a submission over it.
            pass

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, text: str) -> None:
        """Append a submitted prompt, oldest entries dropped past the cap.

        Multiline prompts become one entry per non-empty line, matching what
        the REPL's readline file ends up containing anyway. Consecutive
        duplicates are skipped so re-running a recalled instruction does not
        litter arrow-up with repeats.
        """
        changed = False
        for line in (part.strip() for part in text.splitlines()):
            if not line or (self._entries and line == self._entries[-1]):
                continue
            self._entries.append(line)
            changed = True
        if changed:
            del self._entries[:-MAX_ENTRIES]
            self._save()

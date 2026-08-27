"""Unified-diff rendering for the ``str_replace`` tool.

The engine's generic tool-call formatter collapses ``str_replace``
arguments into a ``repr()`` one-liner, which is useless for reviewing an
edit. This module rebuilds a proper unified diff from the structured
old/new strings, colored the way terminals expect (red deletions, green
additions), and adds word-level highlighting inside each changed line
pair so the exact words that moved stand out without reading both lines
in full.

Every row carries a line-number gutter, which solves two problems at
once: the numbers are the *file's* line numbers (located by reading the
file), and the +/- marker is visually separated from the content — so a
line that itself begins with ``+`` or ``-`` can never be mistaken for,
or swallowed by, the diff chrome.

``str_replace`` events carry no position and no surrounding content, so
:func:`with_file_context` reads the file and pads the edited fragment
with up to :data:`CONTEXT` real lines before and after it; the diff then
shows the change the way ``diff -u`` would — in place, with context.
"""

from __future__ import annotations

import difflib
import os
import re

from rich.style import Style
from rich.text import Text

# Lines of context kept around each change block.
CONTEXT = 3

# Hard cap on rendered diff body lines, so a pathological str_replace
# cannot flood the conversation log.
_MAX_LINES = 400

# Separator between the line-number/marker gutter and the content.
_GUTTER = " │ "

_HEADER = Style(color="yellow", bold=True)
_HUNK = Style(color="cyan", bold=True)
_NUMBER = Style(dim=True)
_CONTEXT = Style(dim=True)
_DEL = Style(color="red")
_ADD = Style(color="green")
_DEL_WORD = Style(color="red", bold=True, bgcolor="grey27")
_ADD_WORD = Style(color="green", bold=True, bgcolor="grey27")

# One run of whitespace, or one run of non-whitespace: word diffs operate
# on whole words while whitespace stays unstyled glue.
_TOKEN_RE = re.compile(r"\s+|\S+")


def _tokens(line: str) -> list[str]:
    return _TOKEN_RE.findall(line)


def _mark_words(source: str, other: str, base: Style, word: Style) -> Text:
    """Render *source* with the words that differ from *other* highlighted.

    Pure insertions or deletions (an empty counterpart) are rendered with
    the plain *base* style — there is nothing to pair words against.
    """
    text = Text(style=base)
    if not source.strip() or not other.strip():
        text.append(source)
        return text
    a, b = _tokens(source), _tokens(other)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        style = None if tag == "equal" else word
        for token in a[i1:i2]:
            text.append(token, style=style)
    return text


def _change_block(dels: list[str], adds: list[str]) -> list[tuple[str, Text]]:
    """Render one ``-``/``+`` block, pairing similar lines for word diffs.

    Returns ``(marker, body)`` pairs; the caller stamps line numbers on.
    """
    out: list[tuple[str, Text]] = []
    # Pair each deleted line with the most similar added line so the word
    # diff highlights the actually-moved words even when line counts shift.
    pair: dict[int, int] = {}
    for i, d in enumerate(dels):
        best_j: int | None = None
        best_ratio = 0.0
        for j, a in enumerate(adds):
            if j in pair.values():
                continue
            ratio = difflib.SequenceMatcher(None, d, a).ratio()
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j
        if best_j is not None:
            pair[i] = best_j
    for i, line in enumerate(dels):
        other = adds[pair[i]] if i in pair else ""
        out.append(("-", _mark_words(line, other, _DEL, _DEL_WORD)))
    rev = {j: i for i, j in pair.items()}
    for j, line in enumerate(adds):
        other = dels[rev[j]] if j in rev else ""
        out.append(("+", _mark_words(line, other, _ADD, _ADD_WORD)))
    return out


def _fmt_range(idx: int, count: int, offset: int) -> str:
    """Hunk-header range for a 0-based *idx* under 1-based *offset*.

    Mirrors ``difflib._format_range_unified``: empty ranges point at the
    line *before* the hunk (``-l,0``).
    """
    begin = idx + offset
    if count == 1:
        return str(begin)
    if count == 0:
        begin -= 1
    return f"{begin},{count}"


def _read_candidates(path: str, cwd: str | None) -> list[str]:
    """Paths to try reading: *path* as given, then resolved against *cwd*."""
    candidates = [path]
    if cwd and not os.path.isabs(path):
        candidates.append(os.path.join(cwd, path))
    return candidates


def _anchor_index(content: str, old: str, new: str) -> int:
    """Index where the edited fragment starts in *content*, -1 if absent.

    Prefers the pre-edit *old* text; falls back to *new* for a file that
    has already been rewritten — the event may be rendered after the tool
    ran, when ``old_str`` no longer exists but its replacement does.
    """
    if old:
        idx = content.find(old)
        if idx >= 0:
            return idx
    if new:
        return content.find(new)
    return -1


def locate_line(path: str, old: str, *, cwd: str | None = None,
                new: str | None = None) -> int:
    """Best-effort 1-based file line where *old* starts, 0 if unknown.

    ``str_replace`` events carry no position, so read the file and find
    the fragment. Relative paths are tried against *cwd* too, since the
    engine may resolve them against its workspace rather than the TUI's
    process directory. *new* (the replacement text) is tried when *old*
    is no longer in the file — the edit already ran.
    """
    if not old and not new:
        return 0
    for cand in _read_candidates(path, cwd):
        try:
            with open(cand, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        idx = _anchor_index(content, old, new or "")
        if idx < 0:
            return 0
        return content.count("\n", 0, idx) + 1
    return 0


def with_file_context(path: str, old: str, new: str,
                      *, cwd: str | None = None) -> tuple[str, str, int] | None:
    """Re-anchor a ``str_replace`` edit inside the real file content.

    Reads *path* (tried against *cwd* for relative paths, like
    :func:`locate_line`), finds the ``old`` fragment, and returns
    ``(old_padded, new_padded, line_offset)`` where the fragment is
    surrounded by up to :data:`CONTEXT` real lines from the file.
    When the file no longer contains ``old`` but contains ``new`` (the
    edit already ran when the event is rendered), the window is anchored
    on ``new`` instead — line numbers stay real either way.
    Returns ``None`` when the file cannot be read or neither fragment
    is found — callers fall back to diffing the bare fragment.
    """
    if not old:
        return None
    for cand in _read_candidates(path, cwd):
        try:
            with open(cand, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        # The tool_call event may be rendered after the tool already ran
        # (the queue drains later than the tool executes), so the file may
        # hold `new` where `old` used to be. Anchor on `new` then: the
        # window and its line numbers are still real, and the pre-edit
        # side of the diff is rebuilt by substituting `old` back in.
        anchor, applied = old, False
        idx = content.find(old)
        if idx < 0:
            idx = content.find(new)
            if idx < 0:
                return None
            anchor, applied = new, True
        lines = content.splitlines(keepends=True)
        start = content.count("\n", 0, idx)
        end = start + anchor.count("\n") + (0 if anchor.endswith("\n") else 1)
        start_ctx = max(start - CONTEXT, 0)
        end_ctx = min(end + CONTEXT, len(lines))
        padded = "".join(lines[start_ctx:end_ctx])
        at = padded.find(anchor)
        old_padded = padded if not applied else padded[:at] + old + padded[at + len(anchor):]
        new_padded = padded if applied else padded[:at] + new + padded[at + len(old):]
        return old_padded, new_padded, start_ctx + 1
    return None


def render_str_replace(path: str, old: str, new: str,
                       context: int = CONTEXT, *,
                       line_offset: int = 1) -> Text:
    """Build a colorized unified diff for one ``str_replace`` edit.

    The diff shows the edited region only — ``str_replace`` carries no
    surrounding file content. *line_offset* is the 1-based file line of
    the first ``old`` line (see :func:`locate_line`); it defaults to 1,
    which numbers the fragment itself.

    Rows are built from ``SequenceMatcher`` opcodes rather than parsing
    ``difflib.unified_diff`` text output: a content line starting with
    ``+``/``-`` must not be able to impersonate (or be eaten by) the
    diff's own markers.
    """
    out = Text()
    for header in (f"--- {path}", f"+++ {path}"):
        out.append(header + "\n", style=_HEADER)

    old_lines = old.splitlines()
    new_lines = new.splitlines()
    # (line number or None for raw chrome, marker, style, body).
    entries: list[tuple[int | None, str, Style, Text]] = []

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines,
                                      autojunk=False)
    for group in matcher.get_grouped_opcodes(context):
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        entries.append((None, "@", _HUNK, Text(
            f"@@ -{_fmt_range(i1, i2 - i1, line_offset)}"
            f" +{_fmt_range(j1, j2 - j1, line_offset)} @@")))
        for tag, a1, a2, b1, b2 in group:
            if tag == "equal":
                for k, ctx_line in enumerate(old_lines[a1:a2]):
                    ctx_body = _mark_words(ctx_line, "", _CONTEXT, _CONTEXT)
                    entries.append(
                        (line_offset + a1 + k, " ", _CONTEXT, ctx_body))
                continue
            dels = old_lines[a1:a2] if tag in ("delete", "replace") else []
            adds = new_lines[b1:b2] if tag in ("insert", "replace") else []
            oi, ni = a1, b1
            for ch_marker, ch_body in _change_block(dels, adds):
                if ch_marker == "-":
                    lineno, oi = line_offset + oi, oi + 1
                else:
                    lineno, ni = line_offset + ni, ni + 1
                base = _DEL if ch_marker == "-" else _ADD
                entries.append((lineno, ch_marker, base, ch_body))

    if not any(marker in "+-" for _n, marker, _s, _b in entries):
        # No line-level change (e.g. only a trailing newline moved).
        note = "no line-level change"
        if old != new:
            note += " (whitespace / trailing newline only)"
        entries.append((None, " ", _CONTEXT, Text(note)))
    if len(entries) > _MAX_LINES:
        dropped = len(entries) - _MAX_LINES
        entries = entries[:_MAX_LINES - 1]
        entries.append((None, " ", _CONTEXT,
                        Text(f"… ({dropped + 1} more lines)")))

    width = max([3] + [len(str(n)) for n, _m, _s, _b in entries
                       if n is not None])
    for raw_num, marker, base, raw_body in entries:
        if raw_num is None:
            # Raw chrome (hunk header, notes): no gutter.
            out.append(raw_body.plain, style=base)
            out.append("\n")
            continue
        line = Text(f"{raw_num:>{width}d} ", style=_NUMBER)
        line.append(marker, style=base)
        line.append(_GUTTER, style=_NUMBER)
        line.append_text(raw_body)
        line.append("\n", style=base)
        out.append_text(line)
    if out.plain.endswith("\n"):
        out.rstrip_end(1)
    return out

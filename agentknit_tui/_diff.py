"""Unified-diff rendering for the ``str_replace`` tool.

The engine's generic tool-call formatter collapses ``str_replace``
arguments into a ``repr()`` one-liner, which is useless for reviewing an
edit. This module rebuilds a proper unified diff from the structured
old/new strings, colored the way terminals expect (red deletions, green
additions), and adds word-level highlighting inside each changed line
pair so the exact words that moved stand out without reading both lines
in full.
"""

from __future__ import annotations

import difflib
import re

from rich.style import Style
from rich.text import Text

# Lines of context kept around each change block.
CONTEXT = 3

# Hard cap on rendered diff body lines, so a pathological str_replace
# cannot flood the conversation log.
_MAX_LINES = 400

_HEADER = Style(color="yellow", bold=True)
_HUNK = Style(color="cyan", bold=True)
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


def _diff_line(prefix: str, line: str, other: str,
               base: Style, word: Style) -> Text:
    body = _mark_words(line, other, base, word)
    return Text.assemble((prefix, base), body)


def _change_block(dels: list[str], adds: list[str]) -> list[Text]:
    """Render one ``-``/``+`` block, pairing similar lines for word diffs."""
    out: list[Text] = []
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
        out.append(_diff_line("-", line, other, _DEL, _DEL_WORD))
    rev = {j: di for di, dj in pair.items() for j in [dj]}
    for j, line in enumerate(adds):
        other = dels[rev[j]] if j in rev else ""
        out.append(_diff_line("+", line, other, _ADD, _ADD_WORD))
    return out


def render_str_replace(path: str, old: str, new: str,
                       context: int = CONTEXT) -> Text:
    """Build a colorized unified diff for one ``str_replace`` edit.

    The diff shows the edited region only — ``str_replace`` carries no
    surrounding file content, so the hunk headers number the lines of the
    replaced string itself.
    """
    out = Text()
    for header in (f"--- {path}", f"+++ {path}"):
        out.append(header + "\n", style=_HEADER)

    old_lines = old.splitlines()
    new_lines = new.splitlines()
    body: list[Text] = []
    dels: list[str] = []
    adds: list[str] = []

    def flush_block() -> None:
        if dels or adds:
            body.extend(_change_block(dels, adds))
            dels.clear()
            adds.clear()

    for line in difflib.unified_diff(old_lines, new_lines, n=context,
                                     lineterm=""):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            flush_block()
            body.append(Text(line + "\n", style=_HUNK))
        elif line.startswith("-"):
            # A deletion after additions starts a fresh change block.
            if adds:
                flush_block()
            dels.append(line[1:])
        elif line.startswith("+"):
            adds.append(line[1:])
        else:
            flush_block()
            body.append(Text(" " + line[1:] + "\n", style=_CONTEXT))
    flush_block()

    if not any(t.plain.startswith(("@@", "-", "+")) for t in body):
        # No line-level change (e.g. only a trailing newline moved).
        note = "no line-level change"
        if old != new:
            note += " (whitespace / trailing newline only)"
        body.append(Text(note + "\n", style=_CONTEXT))
    if len(body) > _MAX_LINES:
        dropped = len(body) - _MAX_LINES
        body = body[:_MAX_LINES]
        body.append(Text(f"… ({dropped} more lines)\n", style=_CONTEXT))

    for chunk in body:
        out.append_text(chunk)
    if out.plain.endswith("\n"):
        out.rstrip_end(1)
    return out

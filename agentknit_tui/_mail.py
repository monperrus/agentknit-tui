"""Bare-email autolinking for assistant messages.

Markdown link syntax (``[name](mailto:x@y)``) and autolinks
(``<x@y>`` / ``<mailto:x@y>``) already carry a ``mailto:`` link style
through Rich's Markdown parser, so terminals render them as clickable
OSC 8 hyperlinks. A *bare* address (``contact x@y now``) does not — the
agent's prose is full of them, and the operator should not have to
copy-paste an address into a mailer.

:func:`linkify_bare_emails` rewrites a Markdown source so every bare
address becomes an inline link to ``mailto:<addr>``, in place. Only the
safe contexts are rewritten:

* already-linked addresses (``[a](mailto:b@c)``, ``<b@c>``) are left
  alone — rewriting them would nest links, which CommonMark forbids;
* addresses inside code spans and fenced/indented code blocks are left
  alone — code is data, and patching a quoted script's string literal
  would corrupt what the operator copies.
"""

from __future__ import annotations

import re

# A pragmatic address: one dotted local part, one dotted domain with a
# 2+ letter TLD. Not RFC 5322 — agent prose holds plain addresses, and
# false positives (``x@y`` without a dot) are not linkified at all.
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])"                      # not glued to a word/local-part char
    r"[A-Za-z0-9._%+-]+"                 # local part
    r"@"
    r"(?:[A-Za-z0-9-]+\.)+"              # domain labels
    r"[A-Za-z]{2,}"                      # TLD
    r"(?![\w.-])"                        # not part of a longer token
)

# Fenced code block (``` or ~~~, optionally with an info string).
_FENCE_RE = re.compile(r"^([ ]{0,3})(`{3,}|~{3,})")

# An already-linked address: inline ``[…](mailto:a@b)`` and autolinks
# ``<a@b>`` / ``<mailto:a@b>``. Skipped so links never nest.
_LINKED_RE = re.compile(
    r"\[[^\]]*\]\(\s*mailto:[^)]+\)"     # [text](mailto:…)
    r"|<mailto:[^>]*>"                   # <mailto:…>
    r"|<[^>\s]*@[^>\s]*>"                # <a@b> autolink
)

# A run of inline code (one or more backticks on both sides).
_CODESPAN_RE = re.compile(r"(`+)[^`]*\1")


def _linkify_line(line: str) -> str:
    """Linkify bare addresses on one non-code-block line."""
    # Split off the parts that must stay verbatim — existing links and
    # inline code — and only rewrite the gaps between them.
    skip = re.compile(f"{_LINKED_RE.pattern}|{_CODESPAN_RE.pattern}")

    def _gap(text: str) -> str:
        return _EMAIL_RE.sub(lambda m: f"[{m.group(0)}](mailto:{m.group(0)})",
                             text)

    out: list[str] = []
    pos = 0
    for m in skip.finditer(line):
        out.append(_gap(line[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_gap(line[pos:]))
    return "".join(out)


def linkify_bare_emails(markdown: str) -> str:
    """Return *markdown* with bare email addresses wrapped as ``mailto:`` links."""
    out: list[str] = []
    fence: str | None = None  # the fence marker that opened a code block
    for line in markdown.splitlines(keepends=True):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(2)
            if fence is None:
                fence = marker[0] * len(marker)
            elif marker.startswith(fence[0]) and len(marker) >= len(fence):
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        # Indented code block: 4+ leading spaces outside a fence.
        if line.startswith("    ") or line.startswith("\t"):
            out.append(line)
            continue
        out.append(_linkify_line(line))
    return "".join(out)

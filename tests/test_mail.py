"""Unit tests for bare-email autolinking in assistant markdown."""

from __future__ import annotations

from agentknit_tui._mail import linkify_bare_emails


def test_bare_address_becomes_mailto_link() -> None:
    out = linkify_bare_emails("Contact martin@example.org now.")
    assert out == ("Contact [martin@example.org](mailto:martin@example.org)"
                   " now.")


def test_multiple_addresses_each_linked() -> None:
    out = linkify_bare_emails("From a@b.co to c-d@e-f.io please")
    assert "[a@b.co](mailto:a@b.co)" in out
    assert "[c-d@e-f.io](mailto:c-d@e-f.io)" in out


def test_existing_inline_link_untouched() -> None:
    src = "Ping [Martin](mailto:martin@example.org) today."
    assert linkify_bare_emails(src) == src


def test_autolink_angle_brackets_untouched() -> None:
    for src in ("Mail <martin@example.org> please.",
                "See <mailto:martin@example.org>."):
        assert linkify_bare_emails(src) == src


def test_mailto_with_headers_untouched() -> None:
    """mailto: URIs with ?subject/&body/&cc pass through verbatim — they are
    already links, so the linkifier must not re-wrap or truncate them."""
    for src in (
        "[mail me](mailto:martin@example.org?subject=Bug%20report&body=Hi%20there)",
        "[cc](mailto:a@b.co?cc=x@y.zz&from=me@mine.io&subject=Hi)",
        "<mailto:martin@example.org?subject=Hello>",
    ):
        assert linkify_bare_emails(src) == src


def test_inline_code_untouched() -> None:
    src = "run `git config user.email a@b.co` first"
    assert linkify_bare_emails(src) == src


def test_fenced_code_block_untouched() -> None:
    src = ("before x@y.zz\n"
           "```python\n"
           "email = 'dev@example.com'\n"
           "```\n"
           "after w@z.zz\n")
    out = linkify_bare_emails(src)
    assert "'dev@example.com'" in out            # code verbatim
    assert "[x@y.zz](mailto:x@y.zz)" in out      # prose linked
    assert "[w@z.zz](mailto:w@z.zz)" in out
    # tilde fences too
    src2 = "~~~\nops@example.net\n~~~\n"
    assert linkify_bare_emails(src2) == src2


def test_indented_code_block_untouched() -> None:
    src = "prose a@b.cc\n\n    code c@d.ee\n"
    out = linkify_bare_emails(src)
    assert "[a@b.cc](mailto:a@b.cc)" in out
    assert "code c@d.ee" in out


def test_trailing_punctuation_not_in_link() -> None:
    out = linkify_bare_emails("write to a@b.co, or not.")
    assert "[a@b.co](mailto:a@b.co)," in out


def test_non_addresses_left_alone() -> None:
    for src in ("no email here",
                "handle @user mention",          # no local part
                "x@y without tld",
                "2+2=4", ):
        assert linkify_bare_emails(src) == src


def test_unterminated_fence_stays_code() -> None:
    src = "```\nops@example.net\n"
    assert linkify_bare_emails(src) == src

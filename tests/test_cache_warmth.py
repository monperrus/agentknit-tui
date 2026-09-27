"""Status-bar cache-warmth countdown (``cache TTL``) in the TUI.

agentknit now reports ``cache_ttl`` / ``cache_expires_in`` on every usage
event; the TUI mirrors them and shows a countdown so a prefix-cache expiry
is a visible event (the next turn re-writes the prompt at full input price)
instead of a silent cost.  These tests pin the pure formatter and the
event→reactive plumbing.
"""

from __future__ import annotations

import math

import pytest

from agentknit_tui.app import _cache_warmth_text


# ── formatter ────────────────────────────────────────────────────────────────

def test_warmth_empty_before_first_cache_proof() -> None:
    assert _cache_warmth_text(300, float("nan")) == ""
    assert _cache_warmth_text(0, 100.0) == ""


def test_warmth_counts_down() -> None:
    assert _cache_warmth_text(300, 250.0) == "cache 4m10s"
    assert _cache_warmth_text(300, 45.0) == "cache 45s"


def test_warmth_expired() -> None:
    assert _cache_warmth_text(300, -1.0) == "cache expired"
    assert _cache_warmth_text(300, 0.0) == "cache expired"


# ── event → reactive plumbing ────────────────────────────────────────────────

def _make_schema() -> dict:
    return {
        "model": "fake-1",
        "endpoint": "https://example.test/v1",
        "inferred_tool_schema": [],
    }


@pytest.mark.asyncio
async def test_usage_event_updates_warmth_reactives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )
        ),
    )

    async with AgentTUI(_make_schema(), non_interactive=True).run_test() as pilot:
        app = pilot.app
        etype, data = "usage", {"prompt": 100, "cache_ttl": 300,
                                "cache_expires_in": 299.5}
        app._apply_event(appmod._QueuedEvent(etype, data),
                         app.query_one("#conversation"))
        await pilot.pause()
        assert app.cache_ttl == 300
        assert math.isclose(app.cache_expires_in, 299.5, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_status_text_includes_warmth_when_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )
        ),
    )

    async with AgentTUI(_make_schema(), non_interactive=True).run_test() as pilot:
        app = pilot.app
        app.cache_ttl = 300
        app.cache_expires_in = 250.0
        await pilot.pause()
        assert "cache 4m10s" in AgentTUI._status_text(app).plain

        app.cache_expires_in = -3.0
        await pilot.pause()
        assert "cache expired" in AgentTUI._status_text(app).plain


@pytest.mark.asyncio
async def test_status_text_omits_warmth_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import agentknit_tui.app as appmod
    from agentknit_tui.app import AgentTUI

    monkeypatch.setattr(
        appmod, "create_client",
        lambda schema: types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )
        ),
    )

    async with AgentTUI(_make_schema(), non_interactive=True).run_test() as pilot:
        app = pilot.app
        await pilot.pause()
        assert "cache " not in AgentTUI._status_text(app).plain

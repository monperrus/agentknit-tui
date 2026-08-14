"""``agentknit-tui`` console entry point.

Usage mirrors the ``agent-glm-5.2.py`` wrappers it replaces:

    agentknit-tui                       # default glm-5.2 via z.ai
    agentknit-tui glm-5.2               # explicit model, default endpoint
    agentknit-tui "qwen3-8b" "https://openrouter.ai/api/v1"
    agentknit-tui --session <id>        # resume a previous trajectory
    agentknit-tui --non-interactive     # drop ask_user* tools from the schema

Wrapper scripts that already build their own schema can import the app
directly::

    from agentknit_tui import AgentTUI
    AgentTUI(schema, non_interactive=True).run()
"""

from __future__ import annotations

import sys

import agentknit
from agentknit import validate_schema
from agentknit.exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    AuthenticationError,
    PricingLimitExceededError,
)

from .app import AgentTUI, build_schema_from_argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    schema, kwargs = build_schema_from_argv(argv)
    try:
        validate_schema(schema)
        agentknit.check_and_display_pricing(schema)
    except AgentSpecDisabledError as exc:
        print(f"Agent disabled: {exc.comment or exc}", file=sys.stderr)
        return 2
    except AgentSpecInvalidError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except PricingLimitExceededError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        return 2

    cache_key = kwargs.pop("cache_key", None)
    app = AgentTUI(schema, **kwargs)
    if cache_key:
        # init_session already ran inside __init__; honour an explicit cache
        # key by overriding the session's cache_key before the first turn.
        app._session["cache_key"] = cache_key  # noqa: SLF001
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

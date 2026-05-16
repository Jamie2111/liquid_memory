"""Command-line entry point for the Liquid Memory gateway.

Installed as the `liquid-memory` console script (see pyproject.toml).
Wraps uvicorn so the install + run path is a single command:

    pip install liquid-memory
    liquid-memory start

Subcommands:
    start    Launch the FastAPI gateway. Reads HYBRID_PROXY_HOST /
             HYBRID_PROXY_PORT env vars (or use --host / --port).
    status   Hit /healthz against a running gateway and print the
             response. Useful for `kubectl exec`-style sanity checks.
    version  Print the installed package version.

Design notes:
    - No subcommand framework dependency (no click / typer). The CLI
      stays in the standard library so `pip install liquid-memory`
      does not pull anything beyond the proxy's existing deps.
    - `start` keeps uvicorn as an optional invocation: power users who
      want gunicorn / hypercorn / their own ASGI server can still do
      `uvicorn liquid_memory.proxy:app` directly. The CLI is the
      convenience path, not the only path.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import __version__


def _cmd_start(args: argparse.Namespace) -> int:
    """Launch the FastAPI gateway via uvicorn."""
    # Lazy import so `liquid-memory --help` does not pay the uvicorn
    # import cost on every invocation.
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover, deps are in install_requires
        sys.stderr.write(
            "uvicorn is required to start the gateway, but the import failed: "
            f"{exc}\nReinstall with `pip install --upgrade liquid-memory`.\n"
        )
        return 1

    host = args.host or os.environ.get("HYBRID_PROXY_HOST", "0.0.0.0")
    port = int(args.port or os.environ.get("HYBRID_PROXY_PORT", "8000"))
    log_level = (args.log_level or os.environ.get("LIQUID_PROXY_LOG_LEVEL", "info")).lower()

    sys.stdout.write(
        f"Liquid Memory v{__version__} gateway starting on http://{host}:{port}\n"
        f"  Point your OpenAI client at: http://{host}:{port}/v1\n"
        f"  Provider keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) must be set in the environment.\n"
        f"  Send SIGINT (Ctrl-C) to stop.\n"
    )

    uvicorn.run(
        "liquid_memory.proxy:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=args.reload,
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Hit /healthz on a running gateway and print the response."""
    try:
        import urllib.error
        import urllib.request
    except ImportError:  # pragma: no cover
        sys.stderr.write("urllib is required but not available; broken Python install?\n")
        return 2

    url = args.url or f"http://{args.host}:{args.port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            sys.stdout.write(f"{resp.status} {url}\n{body}\n")
            return 0
    except urllib.error.URLError as exc:
        sys.stderr.write(f"Could not reach {url}: {exc.reason}\n")
        return 1
    except Exception as exc:  # pragma: no cover, defensive
        sys.stderr.write(f"status check failed: {exc}\n")
        return 2


def _cmd_version(_args: argparse.Namespace) -> int:
    sys.stdout.write(f"liquid-memory {__version__}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser. Exposed so docs / tests can
    introspect the subcommand surface without invoking it."""
    parser = argparse.ArgumentParser(
        prog="liquid-memory",
        description=(
            "Liquid Memory gateway. Lossless prompt compression as a "
            "drop-in OpenAI-compatible proxy. See README for env vars."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"liquid-memory {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Launch the gateway.")
    p_start.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0 or $HYBRID_PROXY_HOST).")
    p_start.add_argument("--port", default=None, help="Bind port (default: 8000 or $HYBRID_PROXY_PORT).")
    p_start.add_argument("--log-level", default=None, help="uvicorn log level (default: info or $LIQUID_PROXY_LOG_LEVEL).")
    p_start.add_argument("--reload", action="store_true", help="Auto-reload on source changes (dev only).")
    p_start.set_defaults(func=_cmd_start)

    # status
    p_status = sub.add_parser("status", help="Health-check a running gateway.")
    p_status.add_argument("--host", default="localhost", help="Host to probe (default: localhost).")
    p_status.add_argument("--port", default=8000, type=int, help="Port to probe (default: 8000).")
    p_status.add_argument("--url", default=None, help="Override full URL instead of host+port.")
    p_status.add_argument("--timeout", default=3.0, type=float, help="HTTP timeout in seconds (default: 3.0).")
    p_status.set_defaults(func=_cmd_status)

    # version (also surfaced via --version, but easier in scripts)
    p_version = sub.add_parser("version", help="Print the installed version.")
    p_version.set_defaults(func=_cmd_version)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

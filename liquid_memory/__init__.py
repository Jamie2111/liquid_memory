"""Liquid Memory, lossless prompt compression as a drop-in OpenAI-compatible proxy.

Public API:

    from liquid_memory import app   # FastAPI app for ASGI servers
    from liquid_memory import cli   # entry point used by the `liquid-memory` command

Most users do not import from this package directly. They install it with
`pip install liquid-memory` (or `docker run liquidmemory/proxy`) and start
the gateway with `liquid-memory start`.
"""

from .proxy import app  # noqa: F401, re-export so `uvicorn liquid_memory:app` works

__version__ = "1.0.0"

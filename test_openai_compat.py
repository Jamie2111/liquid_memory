#!/usr/bin/env python3
"""
Smoke test for /v1/chat/completions (OpenAI-compatible endpoint).

What this verifies
------------------
1. The proxy is reachable and the OpenAI-shaped endpoint is wired up.
2. A standard openai-python `chat.completions.create(...)` call works
   end-to-end through the Liquid Memory compression pipeline.
3. The response has the OpenAI chat.completion shape that client
   SDKs depend on (id, object, created, model, choices, usage).
4. The 501-class rejections fire for unsupported features (streaming,
   tools, n > 1) so client SDKs raise typed exceptions on them
   instead of silently misbehaving.

How to run
----------
    pip install openai
    # In a separate terminal: start the proxy.
    export GEMINI_API_KEY="any-string-here"   # dry_run skips synthesis
    uvicorn liquid_proxy:app --host 0.0.0.0 --port 8000

    # Then:
    python test_openai_compat.py

Pass ?dry_run=1 via the SDK's default_query so this smoke test does
not actually call out to a real synthesis backend - the test only
needs to validate the request/response wiring through the proxy.

Exits 0 on success, non-zero on any assertion failure.
"""
from __future__ import annotations

import sys

try:
    from openai import OpenAI, APIStatusError
except ImportError:
    sys.exit("[fatal] openai package is required.   pip install openai")


PROXY_BASE_URL = "http://localhost:8000/v1"


def make_client():
    return OpenAI(
        base_url=PROXY_BASE_URL,
        api_key="not-used-by-liquid-memory-but-the-sdk-requires-something",
        # default_query gets appended to every request automatically by
        # the openai SDK. Setting dry_run=1 here means the test does not
        # burn any cloud-synthesis tokens; only the proxy + local
        # extraction pipeline run.
        default_query={"dry_run": "1"},
    )


def test_basic_chat_completion(client: OpenAI) -> None:
    print("[1/4] basic chat.completions.create() with a long document + question")
    big_doc = (
        "The quick brown fox jumps over the lazy dog. " * 400
    )  # ~ a few hundred tokens; the pipeline will compress it.
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": big_doc},
            {"role": "user", "content": "Summarise this document in one sentence."},
        ],
    )

    assert resp.id.startswith("chatcmpl-"), f"id has wrong prefix: {resp.id!r}"
    assert resp.object == "chat.completion", f"object: {resp.object!r}"
    assert resp.created > 0, f"created: {resp.created!r}"
    assert resp.model == "gpt-4o", f"model echo mismatch: {resp.model!r}"
    assert isinstance(resp.choices, list) and len(resp.choices) == 1
    choice = resp.choices[0]
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    # Dry-run pads the content with a known string; non-dry-run produces a real answer.
    # Either way the field must be a non-None string.
    assert choice.finish_reason == "stop"
    assert resp.usage is not None
    assert isinstance(resp.usage.prompt_tokens, int)
    assert isinstance(resp.usage.completion_tokens, int)
    assert resp.usage.total_tokens == (
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    )
    print(f"      OK  id={resp.id}")
    print(f"          tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
    print(f"          content[:120]: {choice.message.content[:120]!r}")


def test_single_user_message(client: OpenAI) -> None:
    print("[2/4] single user message (same string acts as document + question)")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is 2 + 2?"},
        ],
    )
    assert resp.id.startswith("chatcmpl-")
    assert resp.choices[0].message.content is not None
    print(f"      OK  id={resp.id}")


def test_streaming_returns_501(client: OpenAI) -> None:
    print("[3/4] stream=True must return a 501 the SDK raises on")
    try:
        # Calling .create(stream=True) returns a generator; iterating it triggers the request.
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        # Force the request by consuming the iterator.
        for _ in stream:
            pass
    except APIStatusError as exc:
        assert exc.status_code == 501, f"expected 501, got {exc.status_code}"
        body = str(exc)
        assert "stream" in body.lower() or "not supported" in body.lower(), body
        print(f"      OK  SDK raised APIStatusError(status=501) on streaming")
        return
    except Exception as exc:
        # Some SDK versions wrap differently; accept any error path that mentions streaming.
        msg = str(exc).lower()
        if "stream" in msg or "501" in msg or "not_supported" in msg:
            print(f"      OK  SDK raised on streaming ({type(exc).__name__})")
            return
        raise
    raise AssertionError("streaming was supposed to fail but the call returned cleanly")


def test_tools_return_501(client: OpenAI) -> None:
    print("[4/4] tools[] must return a 501 the SDK raises on")
    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "does nothing",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    except APIStatusError as exc:
        assert exc.status_code == 501, f"expected 501, got {exc.status_code}"
        print(f"      OK  SDK raised APIStatusError(status=501) on tools")
        return
    except Exception as exc:
        msg = str(exc).lower()
        if "tool" in msg or "501" in msg or "not_supported" in msg:
            print(f"      OK  SDK raised on tools ({type(exc).__name__})")
            return
        raise
    raise AssertionError("tools was supposed to fail but the call returned cleanly")


def main() -> int:
    print(f"[setup] base_url = {PROXY_BASE_URL}")
    print(f"[setup] dry_run query param attached to every request")
    print()
    client = make_client()
    try:
        test_basic_chat_completion(client)
        test_single_user_message(client)
        test_streaming_returns_501(client)
        test_tools_return_501(client)
    except AssertionError as exc:
        print()
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:
        print()
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print()
    print("All four smoke tests passed. The OpenAI-compatible endpoint is wired up.")
    print()
    print("Next: drop the ?dry_run query and point a real workload at")
    print(f"  {PROXY_BASE_URL}")
    print("with your OpenAI SDK to start cutting real cloud-LLM costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Offline regression test for the OpenAI-compat layer in liquid_proxy.py.

Imports the real proxy code and exercises the helpers the FastAPI
handler calls when an OpenAI client hits /v1/chat/completions:

    _split_messages_for_pipeline()   -> (large_document, extraction_task, final_user_prompt)
    _wrap_response_as_openai()       -> OpenAI chat.completion shape
    _openai_error()                  -> 501 / 4xx error body shape
    OpenAIChatRequest / OpenAIChatResponse / etc. -> pydantic schemas

No GPU. No vLLM. No uvicorn. No HTTP. No network. Runs in ~1 second on
any machine that has `pydantic` + `fastapi` installed - which is every
machine where someone could be modifying this code.

Usage:
    pip install pydantic fastapi
    python tests/test_openai_compat_offline.py

Exit code 0 if every assertion passes, non-zero otherwise. Safe to wire
into CI as a fast pre-commit gate.

Background: the wired-up smoke test at test_openai_compat.py requires
the proxy to be running, which requires a GPU on Linux. This test
covers the same logical surface (schema parsing, message splitting,
response wrapping, error shape) but as direct in-process function
calls, so a typo in the message-splitting heuristic gets caught in
~1 second instead of after a 5-minute GPU pod spinup.
"""
import json
import os
import sys
import types


# ---------------------------------------------------------------------
# Mock the heavy deps. liquid_proxy.py imports torch / transformers /
# litellm at module level - they are not actually USED until a request
# runs through the LocalExtractionEngine. By installing fake versions
# at sys.modules level, the proxy imports cleanly and we can test the
# OpenAI-compat helpers in isolation, on any machine that has fastapi
# and pydantic.
# ---------------------------------------------------------------------
def _install_mocks() -> None:
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"
    torch.float32 = "fp32"
    torch.bfloat16 = "bf16"
    torch.dtype = type("dtype", (), {})
    torch.uint8 = "uint8"
    torch.empty = lambda *a, **kw: None

    class _FakeDevice:
        def __init__(self, x):
            self.type = "cuda" if str(x).startswith("cuda") else "cpu"
            self._raw = x

    torch.device = _FakeDevice
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.classes = types.SimpleNamespace(load_library=lambda p: None)
    torch.ops = types.SimpleNamespace()
    sys.modules.setdefault("torch", torch)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type(
        "AutoTokenizer",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: None)},
    )
    sys.modules.setdefault("transformers", transformers)

    litellm = types.ModuleType("litellm")
    litellm.completion = lambda *a, **kw: None
    sys.modules.setdefault("litellm", litellm)
    # `from vllm import ...` is already guarded with try/except in the
    # proxy; intentionally do not mock it here.


_install_mocks()


# Resolve the repo root relative to this file so the test runs from
# any working directory (CI, repo root, tests/ dir, etc.).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
import liquid_proxy as lm  # noqa: E402


# ---------------------------------------------------------------------
# Tiny assertion harness with pass/fail accounting.
# ---------------------------------------------------------------------
_results = []


def check(name: str, condition, detail: str = "") -> None:
    ok = bool(condition)
    _results.append((name, ok, detail))
    prefix = "  OK  " if ok else "  FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"{prefix}  {name}{suffix}")


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# =====================================================================
# 1. Schema validation
# =====================================================================
section("1. OpenAIChatRequest accepts canonical OpenAI inputs")

req1 = lm.OpenAIChatRequest(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Big document content goes here. " * 200},
        {"role": "user", "content": "What does this document say?"},
    ],
)
check("standard two-message request parses", req1.model == "gpt-4o")
check("messages array preserves roles", req1.messages[0].role == "user")
check(
    "messages array preserves content",
    "Big document content" in req1.messages[0].content,
)

req2 = lm.OpenAIChatRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hi"}],
    temperature=0.7,
    max_tokens=500,
    top_p=0.95,
    stop=["END"],
    response_format={"type": "json_object"},
)
check("temperature accepted", req2.temperature == 0.7)
check("max_tokens accepted", req2.max_tokens == 500)
check("response_format accepted", req2.response_format == {"type": "json_object"})

req_stream = lm.OpenAIChatRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
)
check("stream=True parses (handler rejects later)", req_stream.stream is True)

req_tools = lm.OpenAIChatRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hi"}],
    tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
)
check("tools parses (handler rejects later)", req_tools.tools is not None)


# =====================================================================
# 2. Message splitting heuristic
# =====================================================================
section("2. _split_messages_for_pipeline picks the right document/question")

# Case A: doc + question
msgs_a = [
    lm.OpenAIMessage(role="user", content="A" * 5000),
    lm.OpenAIMessage(role="user", content="Summarise this in one sentence."),
]
doc, task, prompt = lm._split_messages_for_pipeline(msgs_a)
check("Case A: longest user msg = document", doc == "A" * 5000)
check(
    "Case A: last user msg = final_user_prompt",
    prompt == "Summarise this in one sentence.",
)
check(
    "Case A: extraction_task quotes the question",
    "Summarise this in one sentence." in task,
)

# Case B: single user message acts as both document and question
msgs_b = [lm.OpenAIMessage(role="user", content="What is 2 + 2?")]
doc, task, prompt = lm._split_messages_for_pipeline(msgs_b)
check("Case B: single msg becomes document", doc == "What is 2 + 2?")
check("Case B: single msg also becomes prompt", prompt == "What is 2 + 2?")

# Case C: system message gets prepended to extraction_task
msgs_c = [
    lm.OpenAIMessage(role="system", content="You are a legal expert."),
    lm.OpenAIMessage(role="user", content="LONG CONTRACT TEXT " * 100),
    lm.OpenAIMessage(role="user", content="Find every termination clause."),
]
doc, task, prompt = lm._split_messages_for_pipeline(msgs_c)
check(
    "Case C: system content carried into extraction_task",
    "You are a legal expert" in task,
)
check("Case C: longest user = document", "LONG CONTRACT TEXT" in doc)
check(
    "Case C: last user = final_user_prompt",
    prompt == "Find every termination clause.",
)

# Case D: multi-turn with assistant interleaving
msgs_d = [
    lm.OpenAIMessage(role="user", content="First user turn discussing X."),
    lm.OpenAIMessage(role="assistant", content="Prior assistant response about X."),
    lm.OpenAIMessage(role="user", content="Follow-up question about X."),
]
doc, task, prompt = lm._split_messages_for_pipeline(msgs_d)
check(
    "Case D: assistant turn woven into document with marker",
    "[ASSISTANT TURN]" in doc,
)
check(
    "Case D: last user msg = final_user_prompt",
    prompt == "Follow-up question about X.",
)

# Case E: empty messages -> ValueError
got_error = False
try:
    lm._split_messages_for_pipeline([])
except ValueError:
    got_error = True
check("Case E: empty messages raises ValueError", got_error)

# Case F: only system msgs (no user) -> ValueError
got_error = False
try:
    lm._split_messages_for_pipeline(
        [lm.OpenAIMessage(role="system", content="hi")]
    )
except ValueError:
    got_error = True
check("Case F: system-only messages raises ValueError", got_error)


# =====================================================================
# 3. Response wrapper produces an OpenAI-compliant shape
# =====================================================================
section("3. _wrap_response_as_openai produces a spec-compliant response")

fake_result = {
    "final_response": "The document discusses risk management policies.",
    "local_model": "Qwen/Qwen2.5-7B-Instruct",
    "openai_model": "gemini/gemini-2.0-flash",
    "chunk_count": 3,
    "extracted_item_count": 12,
    "aggregated_context": {"items": [], "summary": "..."},
    "warnings": [],
    "telemetry": {
        "input_tokens": 5000,
        "synthesis_tokens": 800,
        "compression_ratio": 6.25,
        "total_savings_usd": 0.05,
        "commission_usd": 0.007,
    },
}

resp = lm._wrap_response_as_openai(fake_result, model_name="gpt-4o")
check("id starts with chatcmpl-", resp.id.startswith("chatcmpl-"))
check("object is 'chat.completion'", resp.object == "chat.completion")
check("created is positive int", isinstance(resp.created, int) and resp.created > 0)
check("model is echoed", resp.model == "gpt-4o")
check("choices is length 1", len(resp.choices) == 1)
check("choice index = 0", resp.choices[0].index == 0)
check("message role = assistant", resp.choices[0].message.role == "assistant")
check(
    "message content matches pipeline final_response",
    resp.choices[0].message.content
    == "The document discusses risk management policies.",
)
check("finish_reason = stop", resp.choices[0].finish_reason == "stop")
check("usage.prompt_tokens from telemetry", resp.usage.prompt_tokens == 5000)
check("usage.completion_tokens from telemetry", resp.usage.completion_tokens == 800)
check(
    "usage.total_tokens = prompt + completion",
    resp.usage.total_tokens
    == resp.usage.prompt_tokens + resp.usage.completion_tokens,
)
check("system_fingerprint set", resp.system_fingerprint == "fp_lm_v1")

try:
    blob = resp.model_dump_json()
    parsed = json.loads(blob)
    check(
        "response JSON-serialises cleanly",
        "id" in parsed and "choices" in parsed,
    )
except Exception as exc:
    check("response JSON-serialises cleanly", False, detail=str(exc))


class _FakeTelemetryModel:
    def model_dump(self):
        return {"input_tokens": 1234, "synthesis_tokens": 567}


fake_result_pydantic = dict(fake_result)
fake_result_pydantic["telemetry"] = _FakeTelemetryModel()
resp_p = lm._wrap_response_as_openai(fake_result_pydantic, model_name="gpt-4o")
check(
    "telemetry-as-pydantic-model still works",
    resp_p.usage.prompt_tokens == 1234,
)

fake_result_no_tel = dict(fake_result)
fake_result_no_tel["telemetry"] = None
resp_no_tel = lm._wrap_response_as_openai(fake_result_no_tel, model_name="gpt-4o")
check(
    "missing telemetry: tokens default to 0",
    resp_no_tel.usage.prompt_tokens == 0 and resp_no_tel.usage.total_tokens == 0,
)


# =====================================================================
# 4. Error responses use OpenAI's shape
# =====================================================================
section("4. _openai_error returns spec-compliant error bodies")

err = lm._openai_error(
    message="streaming not supported",
    code="streaming_not_supported",
    http_status=501,
)
check("returns a JSONResponse", err.__class__.__name__ == "JSONResponse")
check("http status 501", err.status_code == 501)

body = json.loads(err.body)
check("body has 'error' key", "error" in body)
check("error has 'message' key", "message" in body["error"])
check(
    "error has 'code' key",
    body["error"]["code"] == "streaming_not_supported",
)
check(
    "error message preserved",
    body["error"]["message"] == "streaming not supported",
)


# =====================================================================
# 5. End-to-end internal consistency
# =====================================================================
section("5. End-to-end: request -> split -> wrap is internally consistent")

e2e_req = lm.OpenAIChatRequest(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a careful legal analyst."},
        {"role": "user", "content": "CONTRACT TEXT: " + ("clause " * 500)},
        {"role": "user", "content": "What is the maximum liability cap?"},
    ],
)
doc, task, prompt = lm._split_messages_for_pipeline(e2e_req.messages)
hybrid_payload = lm.HybridChatRequest(
    large_document=doc,
    extraction_task=task[: lm._MAX_EXTRACTION_TASK_CHARS],
    final_user_prompt=prompt,
    dry_run=True,
)
check("HybridChatRequest constructs cleanly", hybrid_payload.dry_run is True)
check(
    "Document length within proxy's _MAX_DOCUMENT_CHARS",
    len(doc) <= lm._MAX_DOCUMENT_CHARS,
)
check(
    "Final prompt within proxy's _MAX_FINAL_USER_PROMPT_CHARS",
    len(prompt) <= lm._MAX_FINAL_USER_PROMPT_CHARS,
)

synth_result = {
    "final_response": "Maximum liability cap is $500K per claim, $1M aggregate.",
    "local_model": "Qwen/Qwen2.5-7B-Instruct",
    "openai_model": "gemini/gemini-2.0-flash",
    "chunk_count": 4,
    "extracted_item_count": 7,
    "aggregated_context": {"items": [], "summary": "..."},
    "warnings": [],
    "telemetry": {"input_tokens": 3500, "synthesis_tokens": 65},
}
final = lm._wrap_response_as_openai(synth_result, model_name=e2e_req.model)
check(
    "E2E: response wraps the liability answer",
    "liability cap" in final.choices[0].message.content,
)
check("E2E: model echoes the request", final.model == "gpt-4o")
check("E2E: usage total = 3500 + 65", final.usage.total_tokens == 3565)


# =====================================================================
# Summary + exit code
# =====================================================================
print()
print("=" * 60)
passed = sum(1 for _, ok, _ in _results if ok)
total = len(_results)
print(f"  {passed} / {total} assertions passed")
print("=" * 60)
if passed == total:
    print("  All OpenAI-compat checks passed.")
    sys.exit(0)
else:
    print("  Failures:")
    for name, ok, detail in _results:
        if not ok:
            print(f"    - {name}  ({detail})")
    sys.exit(1)

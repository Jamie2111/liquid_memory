# Liquid Memory

**Website:** [liquid-memory.vercel.app](https://liquid-memory.vercel.app)

Liquid Memory cuts the cost of long-context LLM workloads in two completely different ways. Pick the one that matches what you're building.

---

## Two ways to use Liquid Memory

### Mode 1 — The compression proxy

For teams whose app calls a cloud LLM (OpenAI, Anthropic, Gemini, DeepSeek, Llama-as-a-service, etc.) and wants to cut the per-token bill.

- A local Mistral-class model reads and chunks your big document.
- It extracts only the facts that matter for the question, structured as JSON.
- A 5-6x smaller fact pack gets forwarded to your cloud model of choice for synthesis.
- Your existing code changes by one line (the base URL); the response shape stays identical.

Open and runnable from this repo. See **Quickstart** below.

### Mode 2 — The attention library

For teams **training or self-hosting their own LLM** who want linear-time attention so they can fit 128K-token contexts on a single H100 (where standard `MultiheadAttention` OOMs past 32K).

- Drop-in replacement for `torch.nn.MultiheadAttention`. One line in a PyTorch model.
- Parallel mode for prefill + `O(1)`-per-token recurrent decode.
- BF16 in, BF16 out. Precision-safe mode auto-engaged above 64K tokens.
- Compatible with `torch.compile`. Works with Hugging Face Llama / Mistral / Qwen attention by patching `self_attn`.

Licensed separately (compiled kernel + Ed25519 license key). See [`README_INTEGRATION.md`](./README_INTEGRATION.md) for the full integration guide, or email the founders to request access.

---

## Which mode is right for you?

| If you... | Use |
|---|---|
| Build an app that calls OpenAI/Anthropic/Gemini and want a smaller monthly bill | **Mode 1** |
| Train your own LLM and want to extend its context window without buying more GPUs | **Mode 2** |
| Serve your own open-source LLM (Llama, DeepSeek, Mistral, Qwen) and want fewer GPUs for the same context | **Mode 2** |
| Need an air-gapped / fully-offline stack with no cloud LLM calls at all | **Mode 1 in fully-local config**, or **Mode 2** — see the *Fully-local* section below |
| Are an enterprise with regulated data that can't leave your VPC | **Mode 1** (proxy runs in your VPC) or **Mode 2** (library runs inside your training stack) |

---

# Mode 1 — The compression proxy

## The problem

Enterprise teams are getting hit by two problems at once.

First, the **cloud token tax**. Standard RAG and long-context API workflows send raw documents upstream, even when most of the text is boilerplate or filler. A 100-page compliance file, legal packet, or lab report becomes an expensive prompt before any reasoning has happened.

Second, the **privacy radius**. Shipping raw internal documents to an external API expands the blast radius of every request. Sensitive pricing, legal language, customer data, operating procedures, and proprietary research all move further than they need to.

Liquid Memory's proxy changes that flow. It treats the local GPU as a **sieve**, not as the final brain.

## The architecture

### Stage 1: the Liquid Sieve (local)

The first stage runs a local Mistral-class model through **vLLM**. Documents are chunked with overlap, processed with **PagedAttention**, and distilled into compact structured facts. This stage strips away repetition and noise while preserving the high-signal evidence needed for downstream reasoning.

The extraction path is built to fail soft. If the local model returns malformed JSON, Liquid Memory falls back to regex extraction, bracket repair, and raw-evidence recovery blocks so information is never silently dropped.

### Stage 2: the Synthesis call (configurable)

The second stage routes the compressed fact pack through **LiteLLM**. This gives one proxy surface for many synthesis backends without rewriting application code. The target is selected through the `SYNTHESIS_MODEL` env var, so teams can switch between Gemini, OpenAI, Anthropic, DeepSeek, or a local OpenAI-compatible server with a config change instead of an architectural rewrite.

In practice, the pattern looks like this:

1. Local GPU reads the large document.
2. vLLM filters and compresses the content into structured evidence.
3. LiteLLM forwards only the compressed facts to the selected synthesis model.
4. The synthesis model generates the final answer from a far smaller prompt.

## The ROI

Liquid Memory is designed to produce **up to 99% token compression** before the cloud call. A document that would normally consume tens of thousands of remote prompt tokens can be reduced to a compact evidence bundle with only the useful facts preserved.

The local stage is optimized for throughput with **vLLM PagedAttention**, which improves GPU memory efficiency and keeps large-document extraction fast enough for live enterprise workflows. The result is a system that lowers cloud spend, improves privacy posture, and still preserves answer quality by sending the synthesis model only what it needs.

## Prove It On Your Own GPU (≤ 5 min)

Don't take the chart on the website at face value. Reproduce it:

```bash
git clone https://github.com/Jamie2111/liquid_memory.git
cd liquid_memory
pip install -r requirements.txt
huggingface-cli login   # Llama-3.1-8B is gated; accept the license once

# Terminal A - start the proxy (downloads Mistral-7B on first run)
export GEMINI_API_KEY="any-string-here"   # dry_run skips the synthesis call
uvicorn liquid_proxy:app --host 0.0.0.0 --port 8000

# Terminal B - run the benchmark
python benchmark.py
```

[`benchmark.py`](benchmark.py) is a deliberately tight ~50-line script you can read end-to-end in one minute before running it. It loads Llama-3.1-8B-Instruct, generates a 32K-token synthetic document, runs one forward pass on the full document with `torch.cuda.max_memory_allocated()` bookkeeping, POSTs the same document to the local proxy with `dry_run=True`, runs the same forward pass on the compressed fact pack, and prints the VRAM delta. No hidden constants, no telemetry, no dependencies you don't already need.

Output looks like:

```text
[load]  loading meta-llama/Llama-3.1-8B-Instruct
[doc]   building a 32,768-token synthetic document
[run-1] baseline forward pass on the full 32,768-token document
        peak VRAM:  64.43 GB
[proxy] POST http://localhost:8000/v1/hybrid_chat  (dry_run=True, no cloud-synthesis tokens billed)
[run-2] forward pass on the Liquid Memory pack (5,742 tokens)
        peak VRAM:  11.06 GB

>>> 53.37 GB saved (82.8%) <<<
```

### Need a different model, or want to inspect the methodology in detail?

A larger configurable harness lives at [`benchmark/benchmark_vram.py`](benchmark/benchmark_vram.py). It accepts `--model`, `--tokens`, `--baseline-only`, `--json`, and runs on a small non-gated model (`Qwen/Qwen2-0.5B-Instruct`) by default so it works on consumer GPUs. Use it for tuning, CI integration, or regression tracking; use the root `benchmark.py` for the headline proof to send to a prospect.

## Quickstart (cloud synthesis)

The default configuration uses Gemini for synthesis. Any LiteLLM-supported provider works the same way - swap `SYNTHESIS_MODEL` and the matching API key.

### Before you start: one-time Hugging Face setup

The default local extraction model is `mistralai/Mistral-7B-Instruct-v0.3`. Mistral is **Apache 2.0 licensed** (fully open-source - you can use, modify, redistribute, and ship it commercially without restriction), but Hugging Face puts a one-time click-through "Agree and access repository" gate in front of the download. This is a **HF distribution policy, not a license restriction** - once the weights land on your machine, full Apache 2.0 rights apply.

Per-machine one-time setup:

1. Create a Hugging Face account at https://huggingface.co (free, takes 30 seconds).
2. Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and click **"Agree and access repository"** at the top of the page. Approval for Mistral is automatic and typically takes under a minute.
3. Generate a read-scoped access token at https://huggingface.co/settings/tokens → "+ Create new token" → role "Read". Copy the `hf_...` token.
4. On the machine that will run the proxy: `huggingface-cli login` and paste the token when prompted.

After step 4, vLLM can download the Mistral weights on that machine. The auth is cached locally; you only repeat this on a new machine (e.g. a fresh GPU pod).

### Skip the HF gate entirely

If you'd rather not deal with the click-through (e.g. for a smoke test on rented GPU time), point the extractor at a non-gated 7B model. The pipeline does not care which extraction model runs underneath - it cares about token-counts and compression ratios, both of which Qwen2.5 produces equivalently:

```bash
export LOCAL_LLM_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
```

Qwen2.5-7B-Instruct is Apache 2.0 AND ungated on HF. No account, no token, no click-through. Use this path for testing; use Mistral when you want the production default config.

### Stub mode (no GPU required) - for validating the OpenAI-compat layer on a Mac

For developers who want to validate `/v1/chat/completions` wiring without provisioning a GPU at all, the proxy supports a stub-extraction mode. Set:

```bash
export LIQUID_PROXY_STUB_EXTRACTION=1
```

…before starting uvicorn, and the proxy skips both the auth-library load AND the vLLM model load. A `StubExtractionEngine` returns a deterministic fake `aggregated_context` so the OpenAI-compat layer, message-splitting heuristic, response wrapper, and 501 paths all exercise their full code paths.

What this validates:
- Request parsing (OpenAI's `{model, messages, ...}` shape)
- Message splitting (longest user message -> document, last user message -> question)
- Response wrapping (the OpenAI `chat.completion` shape with id / object / created / choices / usage)
- 501 paths for `stream=True`, `tools=[...]`, `n > 1`

What this does NOT validate:
- Real document compression (nothing is actually compressed)
- vLLM correctness on the customer's hardware
- Cloud-LLM round-trip (use `?dry_run=1` to skip synthesis too)

Recommended for local development, CI smoke tests, and validating the OpenAI-compat surface before deploying. **Do not use in production.** Install requirements (skipping vllm, which is Linux+CUDA-only):

```bash
pip install fastapi uvicorn litellm requests openai torch transformers pydantic python-dotenv

# Stub mode + dry_run = pure OpenAI-compat wiring test
export LIQUID_PROXY_STUB_EXTRACTION=1
export GEMINI_API_KEY="any-string"
uvicorn liquid_proxy:app --port 8000 &
python test_openai_compat.py    # all four asserts should pass in ~3 seconds
```

### Start the proxy

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export SYNTHESIS_MODEL="gemini/gemini-3.1-flash-lite"

uvicorn liquid_proxy:app --host 0.0.0.0 --port 8000
```

The proxy now exposes two POST endpoints. Pick the one that matches the integration shape you want.

### `/v1/chat/completions`  -  OpenAI-compatible (recommended)

Point any OpenAI client (`openai-python`, `openai-node`, LangChain's `ChatOpenAI`, LlamaIndex, etc.) at `http://localhost:8000/v1` and call `chat.completions.create(...)` as you would against api.openai.com. The proxy decides which `messages` entry is the "large document" (longest user message) and which is the "question" (last user message), runs the local extraction + compression pipeline, forwards a smaller fact pack to the synthesis backend selected by `SYNTHESIS_MODEL`, and returns a standard OpenAI chat.completion response.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": large_document_text},
        {"role": "user", "content": "Summarise the key risks."},
    ],
)
print(resp.choices[0].message.content)
```

That's the literal "change one line of code" integration. No restructuring of your messages array, no new SDK to learn. Existing tooling that speaks the OpenAI chat API (eval frameworks, prompt managers, observability sidecars) keeps working.

**v1 limitations.** The following OpenAI features return a clean 501 with an OpenAI-shaped error body so client SDKs raise typed exceptions on them rather than silently misbehaving:

- `stream=True`
- `tools` / `functions`
- `n > 1`

Use `/v1/hybrid_chat` below for those cases until v2 ships streaming + tool-calling support.

**Smoke test.** A 4-test smoke script ships at the repo root:

```bash
pip install openai
python test_openai_compat.py
```

Validates: basic request/response shape, single-message handling, the 501 path for streaming, and the 501 path for tools. Uses `?dry_run=1` internally so the test does not burn any cloud-synthesis tokens.

### `/v1/hybrid_chat`  -  explicit three-input form

For workloads that want to control extraction-task wording, document/question split, and dry-run behavior explicitly:

```bash
curl -X POST http://127.0.0.1:8000/v1/hybrid_chat \
  -H "Content-Type: application/json" \
  --data '{
    "large_document": "...",
    "extraction_task": "Extract every quantitative fact and named entity.",
    "final_user_prompt": "Summarise the key risks in this document."
  }'
```

The OpenAI-compatible endpoint delegates to this same pipeline internally - the only difference is the request/response shape.

## Fully-local stack (Mistral extraction + DeepSeek / Llama / Qwen synthesis)

For air-gapped, regulated, or cost-sensitive deployments you can run **everything** locally - no cloud LLM calls at any stage. The proxy uses LiteLLM's OpenAI-compatible adapter, so any local server that speaks the OpenAI chat-completions schema works as a synthesis backend. The most common patterns:

### Pattern A: Mistral + DeepSeek-V3 (both via vLLM, on the same machine or different GPUs)

```bash
# Terminal 1: serve DeepSeek-V3 as an OpenAI-compatible endpoint on port 9000
vllm serve deepseek-ai/DeepSeek-V3 \
  --host 0.0.0.0 --port 9000 \
  --max-model-len 32768

# Terminal 2: start the Liquid Memory proxy, pointing synthesis at the local DeepSeek
export OPENAI_API_KEY="not-used-but-litellm-requires-the-var"
export OPENAI_API_BASE="http://localhost:9000/v1"
export SYNTHESIS_MODEL="openai/deepseek-ai/DeepSeek-V3"
uvicorn liquid_proxy:app --host 0.0.0.0 --port 8000
```

Now the proxy reads the document with Mistral (locally), extracts a compressed fact pack, and forwards the pack to DeepSeek (locally). Nothing leaves the host.

### Pattern B: Mistral + Llama-3.1-70B (via TGI or vLLM)

```bash
# Serve Llama-3.1-70B with text-generation-inference
docker run --gpus all -p 9000:80 ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-70B-Instruct

# Point Liquid Memory at it
export OPENAI_API_BASE="http://localhost:9000/v1"
export SYNTHESIS_MODEL="openai/meta-llama/Llama-3.1-70B-Instruct"
```

### Pattern C: Mistral + Qwen2.5 (lighter footprint, single 80GB GPU)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 9000 --max-model-len 32768
export OPENAI_API_BASE="http://localhost:9000/v1"
export SYNTHESIS_MODEL="openai/Qwen/Qwen2.5-7B-Instruct"
```

### When fully-local makes sense

- **Regulated industries** (defense, healthcare, finance, legal) where prompts contain controlled data that cannot leave the perimeter.
- **High-volume internal workloads** where the per-token cloud rate dominates total cost; running on amortised hardware can be cheaper at scale.
- **Sovereign-cloud deployments** in jurisdictions where US/EU LLM APIs are restricted.
- **Reproducibility-bound workflows** (research, audit, regulatory submissions) where model drift on a hosted endpoint is unacceptable.

The trade-off is GPU footprint: you need enough VRAM to host both the extraction model (Mistral-7B, ~14 GB) and the synthesis model (DeepSeek-V3 needs a lot more; Llama-3.1-70B around 140 GB unquantised; Qwen2.5-7B around 14 GB). Pattern C fits on a single 80 GB H100; Pattern A typically needs a multi-GPU node.

## Repository layout

```text
liquid_memory/
├── benchmark.py                 # 50-line zero-trust VRAM proof (Llama-3.1-8B)
├── liquid_proxy.py              # FastAPI proxy (Mode 1)
├── benchmark/
│   └── benchmark_vram.py        # Configurable harness (--model, --tokens, etc.)
├── dist_public/                 # AOT artifacts (Mode 2, license-gated)
│   ├── LiquidMemory_AOTI.pt2
│   ├── liquid_memory_auth.so
│   ├── AOTI_METADATA.json
│   └── MANIFEST.md
├── README.md                    # this file
├── README_INTEGRATION.md        # Mode 2 - attention-library integration guide
├── requirements.txt
└── .gitignore
```

---

# Mode 2 — The attention library

A drop-in replacement for `torch.nn.MultiheadAttention` with linear-time scaling and stable execution to 128K tokens.

```python
from liquid_memory import LiquidMemory

# In your Transformer block, swap:
#   self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
# for:
self.attn = LiquidMemory(d_model, n_heads, batch_first=True)

# Call exactly as you would nn.MultiheadAttention:
out, _ = self.attn(x, x, x, is_causal=True)
```

That's the whole change. Forward shapes, return types, and masking semantics (`is_causal`, `key_padding_mask`) are preserved.

For autoregressive decode there's an additional `step(token)` method that runs in `O(1)` per token regardless of context length - the recurrent hidden state lives on the module and persists across calls.

## What you get

- **Linear time and memory.** 128K-token contexts fit on a single H100 in BF16; vanilla `MultiheadAttention` OOMs at ~32K.
- **Per-token decode.** `O(1)` step instead of `O(N)` per generation step. Long-running chat sessions stay cheap.
- **`torch.compile` compatible.** The kernel registers as a custom op and gets treated as a black box; surrounding PyTorch fuses normally.
- **Hugging Face patch path.** A two-screen recipe patches `LlamaAttention` (or any architecture-specific attention class) cleanly. See [`README_INTEGRATION.md` § 3.2](./README_INTEGRATION.md).

## Use cases

- **Long-context training.** Teams building a foundation model who want to extend context to 128K without quadratic VRAM blow-up. Train at full sequence length on the GPUs you already have instead of buying more.
- **Self-hosted inference.** Teams serving Llama-3.1-70B, Qwen2.5, Mistral, DeepSeek, etc. on their own hardware. Patch `self_attn` once, serve 5x longer contexts per GPU.
- **Research / ablations.** Use Liquid Memory as a non-quadratic baseline in long-context experiments.
- **On-prem enterprise LLMs.** Regulated industries running their own model behind the firewall who want longer contexts without scaling out the cluster.

## Built-in benchmark

The module ships with a one-shot comparison harness:

```python
from liquid_memory import LiquidMemory
import torch

LiquidMemory.benchmark(
    embed_dim=4096,
    num_heads=32,
    batch_size=1,
    seq_lengths=(2048, 8192, 32_768, 131_072),
    dtype=torch.bfloat16,
    device="cuda",
)
```

Sample output (H100, BF16):

```text
 seq_len │     LM mem │     TX mem │    LM ms │    TX ms │  speedup
─────────────────────────────────────────────────────────────────────
    2048 │     0.21GB │     0.34GB │      1.2 │      2.7 │     2.3x
    8192 │     0.84GB │     5.12GB │      4.7 │     41.3 │     8.8x
   32768 │     3.36GB │       OOM  │     18.6 │     OOM  │      n/a
  131072 │    13.42GB │       OOM  │     74.2 │     OOM  │      n/a
```

## Licensing

The compiled kernel and the `liquid_memory.py` wrapper module are distributed under a commercial license. The compiled artifacts in `dist_public/` are publicly readable so you can verify SHA-256 against `MANIFEST.md` before installation, but executing the kernel requires a provisioned Ed25519 license key set via `LM_PRIVATE_KEY` at process start.

To request a license / pilot, email the founders (see Contact below).

---

## Contact

- jamieobala2028@u.northwestern.edu
- selinasun2028@u.northwestern.edu

For security issues: same emails; we acknowledge within one business day.

---

## Why this wins

Liquid Memory is not another chatbot wrapper. It is cost-control and privacy infrastructure for AI teams that want to push the long-context frontier without paying for noise.

That makes it an unusually broad wedge - it sells into TWO completely different buyers:

1. **App developers** burning $20k+/mo on OpenAI/Anthropic long-context calls. They install the proxy (Mode 1) and the bill drops in one billing cycle.
2. **Foundation-model teams and self-hosters** paying for the GPUs that quadratic attention forces them to over-provision. They patch one line of PyTorch (Mode 2) and serve 4x longer contexts on the same cluster.

Local GPUs do the filtering. Synthesis models - cloud or local - do the reasoning. The business stops paying for noise.

---

© 2026 Liquid Memory, Inc. Distribution and reverse engineering of the compiled kernel are governed by your license agreement.

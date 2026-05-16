#!/usr/bin/env python3
# =====================================================================
# Liquid Memory - reproducible VRAM benchmark.
# ---------------------------------------------------------------------
# This script reproduces the headline VRAM-savings claim from
#   https://liquid-memory.vercel.app/#proof
# on the developer's own GPU, with their own eyes, in their own
# terminal. No marketing copy: just `python benchmark_vram.py`.
#
# WHAT IT MEASURES
# ----------------
# Same downstream model, same task, two input paths:
#
#   1. Baseline.        Feed the original 32K-token document to the
#                       model. Run one forward pass with use_cache=False
#                       so the attention path matches the O(N^2) blowup
#                       you would see in a naive long-context call.
#                       Record peak device memory.
#
#   2. Liquid Memory.   Send the same document to the local Liquid
#                       Memory proxy (POST /v1/hybrid_chat, dry_run=True
#                       so no cloud-synthesis tokens are billed). The
#                       proxy returns an aggregated_context JSON pack
#                       built by chunked extraction. Feed THAT to the
#                       same downstream model, record peak device
#                       memory.
#
# The script prints:
#   - peak memory for each run
#   - absolute and percentage VRAM reduction
#   - the proxy-reported compression ratio (input / compressed tokens)
#
# WHY THIS IS THE RIGHT BENCHMARK
# -------------------------------
# The chart at liquid-memory.vercel.app/#proof plots peak VRAM vs
# input sequence length. The O(N^2) claim is about the attention
# compute / KV-cache footprint of the *downstream* model that has
# to process the prompt. Liquid Memory does not change that model;
# it changes what reaches it. By holding the model fixed and varying
# only the input length (full vs compressed), we isolate the savings
# the website actually claims.
#
# NOT A PRODUCTION HARNESS
# ------------------------
# This is a proof script, not a perf-regression suite. It runs one
# forward pass per side, on one synthetic document, without warmup
# or repeats. Use it to convince yourself the math works; use a
# proper harness (e.g. text-generation-inference benchmarks, or
# vLLM's benchmark_throughput.py) for production capacity planning.
#
# QUICKSTART
# ----------
#   # 1. Install deps. torch + transformers cover the downstream-
#   #    model half; requests is for hitting the local proxy.
#   pip install torch transformers requests
#
#   # 2. Start the Liquid Memory proxy in a separate terminal.
#   #    (uvicorn loads Mistral-7B locally; first run downloads ~14 GB.)
#   export GEMINI_API_KEY="..."   # any provider key; dry_run skips synthesis
#   uvicorn liquid_memory.proxy:app --host 0.0.0.0 --port 8000
#
#   # 3. Run the benchmark.
#   python benchmark/benchmark_vram.py
#
# OUTPUT IS FREE AND DETERMINISTIC. Anyone can run this. That is
# the entire point.
# =====================================================================

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------
# Hard deps are imported lazily inside _import_deps() so `--help` and
# argparse errors render without needing torch / transformers installed.
# The actual run fails early with a clear message naming the missing
# package.
# ---------------------------------------------------------------------
torch = None  # populated by _import_deps()
AutoModelForCausalLM = None
AutoTokenizer = None
requests = None


def _import_deps(require_requests: bool) -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, requests
    try:
        import torch as _torch  # noqa: F401
        torch = _torch
    except ImportError:
        sys.exit("[fatal] `torch` is required.   pip install torch")
    try:
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
        )
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
    except ImportError:
        sys.exit("[fatal] `transformers` is required.   pip install transformers")
    if require_requests:
        try:
            import requests as _requests
            requests = _requests
        except ImportError:
            sys.exit(
                "[fatal] `requests` is required to talk to the proxy. "
                "Install it (pip install requests) or pass --baseline-only "
                "to skip the Liquid Memory side."
            )


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
# 32K is the chart's anchor; HF baseline OOMs past this on A100 80GB.
DEFAULT_TOKENS = 32_768
DEFAULT_PROXY_URL = "http://localhost:8000/v1/hybrid_chat"
# The proxy's local extraction step on Mistral-7B can take a couple
# of minutes on a 32K-token document. Give it room.
DEFAULT_PROXY_TIMEOUT_S = 600


# ---------------------------------------------------------------------
# Device + memory helpers
# ---------------------------------------------------------------------
def detect_device() -> Tuple[str, str]:
    """Return (torch_device_str, friendly_label) for logging."""
    if torch.cuda.is_available():
        return "cuda", f"CUDA ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS, unified memory)"
    return "cpu", "CPU (no GPU detected - the headline VRAM claim does not apply)"


def reset_peak_memory(device: str) -> None:
    """Drop caches + zero the peak-memory counter before each pass."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        # MPS has no explicit peak-stat reset; emptying the cache and
        # noting the current allocated bytes before/after is the best
        # approximation available.
        torch.mps.empty_cache()


def peak_memory_bytes(device: str) -> int:
    """
    Return peak memory used since the last reset. On CUDA this is the
    real peak. On MPS we return current_allocated_memory(), which is
    a snapshot rather than a peak; close enough for one forward pass
    where the post-pass tensors are still resident. On CPU we report
    process RSS via psutil if available.
    """
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    if device == "mps":
        return torch.mps.current_allocated_memory()
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return -1


def fmt_bytes(n: int) -> str:
    if n is None or n < 0:
        return "n/a"
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024:.1f} KB"


# ---------------------------------------------------------------------
# Synthetic document
# ---------------------------------------------------------------------
# We build a deterministic synthetic document that tokenises to
# approximately --tokens tokens. The structure (numbered facts with
# repeated boilerplate around them) gives the local extraction model
# something real to compress while keeping the benchmark fully
# reproducible across runs and machines.
# ---------------------------------------------------------------------
def generate_synthetic_document(tokenizer, target_tokens: int) -> str:
    boilerplate = (
        "Section {i}. Pursuant to the policy outlined in the previous "
        "section, the following operational guidance applies to all "
        "covered transactions during the reporting period. Standard "
        "definitions remain in effect unless explicitly superseded. "
    )
    fact = (
        "Recorded fact {i}: counterparty number {i} executed {j} "
        "units of transaction class {k} at a unit price of ${p}. "
    )
    chunks: list[str] = []
    i = 0
    while True:
        i += 1
        chunks.append(boilerplate.format(i=i))
        chunks.append(fact.format(
            i=i, j=(i * 7) % 1000, k=("alpha", "beta", "gamma")[i % 3],
            p=round(10 + (i * 1.37) % 990, 2),
        ))
        text = "".join(chunks)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_tokens:
            # Trim to exactly target_tokens for deterministic comparison
            ids = ids[:target_tokens]
            return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------
# One forward pass with peak-memory bookkeeping
# ---------------------------------------------------------------------
def run_forward_pass(model, tokenizer, text: str, device: str) -> Tuple[int, float, int]:
    """
    Run a single forward pass on `text`. Returns
    (peak_memory_bytes, elapsed_seconds, input_token_count).
    """
    reset_peak_memory(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    n_tokens = int(input_ids.shape[-1])
    t0 = time.time()
    with torch.inference_mode():
        # use_cache=False forces the full O(N^2) attention path that
        # the chart's HF baseline reflects. With caching on, the peak
        # memory is dominated by the KV cache instead of the compute
        # buffer; both are valid metrics but the chart uses compute.
        _ = model(input_ids=input_ids, use_cache=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return peak_memory_bytes(device), elapsed, n_tokens


# ---------------------------------------------------------------------
# Liquid Memory proxy call
# ---------------------------------------------------------------------
def compress_via_proxy(
    document: str, proxy_url: str, timeout_s: int
) -> Dict[str, Any]:
    """
    Hit POST /v1/hybrid_chat on the local proxy with dry_run=True.
    Returns the parsed response dict. Raises SystemExit with clear
    instructions if the proxy is unreachable.
    """
    payload = {
        "large_document": document,
        "extraction_task": (
            "Extract every quantitative fact, named entity, monetary "
            "figure, and structural section from the document. Preserve "
            "evidence quotes verbatim. Do not paraphrase or summarise."
        ),
        "final_user_prompt": (
            "Return the compressed fact pack only. A downstream "
            "benchmark harness will consume aggregated_context "
            "directly, no synthesis is required."
        ),
        # dry_run skips the cloud synthesis call so the benchmark
        # does not burn provider credits. Local extraction still runs.
        "dry_run": True,
    }

    try:
        r = requests.post(proxy_url, json=payload, timeout=timeout_s)
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"\n[fatal] could not reach Liquid Memory proxy at {proxy_url}\n\n"
            "Start the proxy in a separate terminal:\n"
            "    export GEMINI_API_KEY=...   # any provider key; dry_run skips synthesis\n"
            "    uvicorn liquid_memory.proxy:app --host 0.0.0.0 --port 8000\n\n"
            "Then re-run this script. Pass --baseline-only to skip the\n"
            "Liquid Memory side entirely (you'll only see the O(N^2)\n"
            "baseline number).\n"
        )

    if not r.ok:
        sys.exit(
            f"[fatal] proxy returned HTTP {r.status_code}:\n"
            f"        {r.text[:500]}"
        )
    return r.json()


def serialise_aggregated_context(aggregated_context: Dict[str, Any]) -> str:
    """
    Convert the proxy's aggregated_context dict into the same JSON
    string the proxy itself feeds to the cloud-synthesis model
    (see liquid_proxy._build_synthesis_messages). This makes the
    benchmark's "compressed input" exactly what the production path
    would deliver downstream, so the VRAM measurement is honest.
    """
    return json.dumps(aggregated_context, indent=2, ensure_ascii=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproducible VRAM benchmark for Liquid Memory."
    )
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                    help=f"Target input token count (default: {DEFAULT_TOKENS}).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id for the downstream pass (default: {DEFAULT_MODEL}).")
    ap.add_argument("--proxy", default=DEFAULT_PROXY_URL,
                    help=f"Liquid Memory proxy URL (default: {DEFAULT_PROXY_URL}).")
    ap.add_argument("--proxy-timeout", type=int, default=DEFAULT_PROXY_TIMEOUT_S,
                    help="Seconds to wait for the proxy to compress. "
                         "Long because Mistral-7B extraction takes a while.")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Skip the proxy call. Useful if you just want to see "
                         "what the raw O(N^2) baseline looks like on your GPU.")
    ap.add_argument("--json", action="store_true",
                    help="Emit a final JSON blob in addition to the human "
                         "log so CI / scripts can consume the numbers.")
    args = ap.parse_args()

    # Now actually import torch / transformers (and requests if needed).
    # Deferred so `--help` and arg-parsing errors render without
    # requiring the heavy ML stack to be installed.
    _import_deps(require_requests=not args.baseline_only)

    device, device_label = detect_device()
    print()
    print("============================================================")
    print("  Liquid Memory - VRAM benchmark")
    print("============================================================")
    print(f"  device : {device_label}")
    print(f"  model  : {args.model}")
    print(f"  target : {args.tokens:,} tokens")
    if not args.baseline_only:
        print(f"  proxy  : {args.proxy}")
    print()

    # ── Step 1. Load the downstream model. -----------------------------
    print("[load]   downloading + loading downstream model from HF Hub ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()

    # ── Step 2. Build the input document. ------------------------------
    print(f"[build]  generating a synthetic ~{args.tokens:,}-token document ...")
    document = generate_synthetic_document(tokenizer, args.tokens)
    doc_tokens = len(tokenizer(document, add_special_tokens=False)["input_ids"])
    print(f"         {doc_tokens:,} tokens, {len(document):,} characters")

    # ── Step 3. Baseline forward pass. ---------------------------------
    print()
    print("[run-1]  baseline: forward pass on the full document")
    baseline_peak, baseline_secs, baseline_n = run_forward_pass(
        model, tokenizer, document, device
    )
    print(f"         {baseline_n:,} tokens in, "
          f"peak memory {fmt_bytes(baseline_peak)}, "
          f"{baseline_secs:.2f}s")

    # ── Step 4. Liquid Memory compression + forward pass. -------------
    lm_peak: Optional[int] = None
    lm_secs: Optional[float] = None
    lm_n: Optional[int] = None
    compress_secs: Optional[float] = None
    proxy_ratio: Optional[float] = None
    proxy_telemetry: Optional[Dict[str, Any]] = None

    if not args.baseline_only:
        print()
        print(f"[run-2]  compressing via Liquid Memory proxy ...")
        t0 = time.time()
        proxy_resp = compress_via_proxy(document, args.proxy, args.proxy_timeout)
        compress_secs = time.time() - t0
        proxy_telemetry = proxy_resp.get("telemetry") or {}
        proxy_ratio = proxy_telemetry.get("compression_ratio")
        compressed_pack = serialise_aggregated_context(
            proxy_resp.get("aggregated_context", {})
        )
        pack_tokens = len(
            tokenizer(compressed_pack, add_special_tokens=False)["input_ids"]
        )
        print(f"         compressed in {compress_secs:.1f}s "
              f"(local_model={proxy_resp.get('local_model','?')})")
        print(f"         compressed pack: {pack_tokens:,} tokens "
              f"({doc_tokens / max(pack_tokens, 1):.1f}x reduction by re-tokenisation; "
              f"proxy reports {proxy_ratio})")

        print()
        print("[run-3]  forward pass on the compressed fact pack")
        lm_peak, lm_secs, lm_n = run_forward_pass(
            model, tokenizer, compressed_pack, device
        )
        print(f"         {lm_n:,} tokens in, "
              f"peak memory {fmt_bytes(lm_peak)}, "
              f"{lm_secs:.2f}s")

    # ── Step 5. Result table. ------------------------------------------
    print()
    print("============================================================")
    print("  RESULT")
    print("============================================================")
    print(f"  Baseline peak VRAM : {fmt_bytes(baseline_peak)}")
    if lm_peak is not None:
        savings = baseline_peak - lm_peak
        pct = (100.0 * savings / baseline_peak) if baseline_peak > 0 else 0.0
        print(f"  Liquid Memory peak : {fmt_bytes(lm_peak)}")
        print(f"  Saved              : {fmt_bytes(savings)}  ({pct:.1f}%)")
        if proxy_ratio:
            print(f"  Proxy compression  : {proxy_ratio}x  (input / compressed tokens)")
    print("============================================================")
    print()
    if device == "cpu":
        print("NOTE: you ran on CPU, so 'VRAM' above is process RSS, not GPU memory.")
        print("      The qualitative O(N^2) -> O(N) shape still holds; absolute numbers")
        print("      will be much smaller than the GPU plot at liquid-memory.vercel.app.")
        print()

    if args.json:
        out = {
            "device": device,
            "device_label": device_label,
            "model": args.model,
            "target_tokens": args.tokens,
            "baseline": {
                "input_tokens": baseline_n,
                "peak_bytes": baseline_peak,
                "latency_s": baseline_secs,
            },
            "liquid_memory": None if lm_peak is None else {
                "input_tokens": lm_n,
                "peak_bytes": lm_peak,
                "latency_s": lm_secs,
                "compression_seconds": compress_secs,
                "proxy_compression_ratio": proxy_ratio,
                "proxy_telemetry": proxy_telemetry,
            },
        }
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

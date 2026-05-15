#!/usr/bin/env python3
"""
Liquid Memory · zero-trust VRAM proof.

Reproduces the headline chart from liquid-memory.vercel.app/#proof
on your own hardware in under 5 minutes. No marketing copy, no
hidden constants: read the script, run the script, see the number.

Setup
-----
    pip install torch transformers requests
    huggingface-cli login   # Llama-3.1-8B is gated; accept the
                            # license at huggingface.co/meta-llama/Llama-3.1-8B-Instruct

In a SEPARATE terminal, start the Liquid Memory proxy first:
    export GEMINI_API_KEY="any-string"   # dry_run skips synthesis
    uvicorn liquid_proxy:app --host 0.0.0.0 --port 8000

Then run this script:
    python benchmark.py

For a smaller / non-gated model, or to dig into the methodology,
see the configurable harness at benchmark/benchmark_vram.py.
"""
import json, requests, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROXY = "http://localhost:8000/v1/hybrid_chat"
N_TOKENS = 32_768

print(f"[load]  loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

print(f"[doc]   building a {N_TOKENS:,}-token synthetic document")
doc = ("The quick brown fox jumps over the lazy dog. " * 8000)[: N_TOKENS * 4]
ids = tok(doc, return_tensors="pt", add_special_tokens=False).input_ids[:, :N_TOKENS].cuda()

def measure_peak_vram(input_ids):
    """One forward pass with peak-memory bookkeeping."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)   # use_cache=False forces full O(N^2) path
    return torch.cuda.max_memory_allocated()

print(f"[run-1] baseline forward pass on the full {N_TOKENS:,}-token document")
baseline = measure_peak_vram(ids)
print(f"        peak VRAM: {baseline / 1e9:6.2f} GB")

print(f"[proxy] POST {PROXY}  (dry_run=True, no cloud-synthesis tokens billed)")
resp = requests.post(PROXY, json={
    "large_document": doc,
    "extraction_task": "Extract every quantitative fact and named entity.",
    "final_user_prompt": "Return only the compressed fact pack.",
    "dry_run": True,
}, timeout=600).json()
pack = json.dumps(resp["aggregated_context"])
pack_ids = tok(pack, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

print(f"[run-2] forward pass on the Liquid Memory pack ({pack_ids.shape[1]:,} tokens)")
liquid = measure_peak_vram(pack_ids)
print(f"        peak VRAM: {liquid / 1e9:6.2f} GB")

saved = baseline - liquid
print(f"\n>>> {saved / 1e9:.2f} GB saved ({100 * saved / baseline:.1f}%) <<<")

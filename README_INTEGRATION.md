# Liquid Memory — Integration Guide

A drop-in replacement for `torch.nn.MultiheadAttention` with linear-time
scaling and stable execution to 128k tokens. This guide walks a Transformer
codebase through the swap in under ten minutes.

---

## TL;DR — three lines

```python
# 1. Import
from liquid_memory import LiquidMemory

# 2. Swap the constructor (signature is identical for the common case)
self.attn = LiquidMemory(embed_dim=d_model, num_heads=n_heads, batch_first=True)

# 3. Call it exactly as you would nn.MultiheadAttention
out, _ = self.attn(x, x, x, is_causal=True)
```

That's it. The rest of this document covers provisioning, advanced
configuration, and migration of specific frameworks (Hugging Face
Transformers, decoder loops, long-context training).

---

## 1. Installation

```bash
pip install torch>=2.8
```

Place the Liquid Memory distribution in your project:

```
your_project/
├── liquid_memory.py            # the MultiheadAttention shim
├── liquid_memory_loader.py     # picks the .pt2 matching the host GPU
└── dist_public/
    ├── LiquidMemory_AOTI_sm_89_L2048_trained.pt2     # RTX 40-series, L40
    ├── LiquidMemory_AOTI_sm_90_L2048_trained.pt2     # H100, H200
    ├── LiquidMemory_AOTI_sm_90_L4096_trained.pt2
    ├── LiquidMemory_AOTI_sm_90_L8192_trained.pt2
    ├── LiquidMemory_AOTI_sm_90_L16384_trained.pt2
    ├── LiquidMemory_AOTI_sm_90_L32768_trained.pt2
    └── LiquidMemory_AOTI_sm_90_L65536_trained.pt2    # the long-context artifact
```

Each `.pt2` is compiled for a specific GPU compute capability and a specific
sequence length. The loader picks the right one at runtime based on the host
GPU and the input shape. If your deployment target is missing from the
shipped set, contact the Liquid Memory team to request a build for that arch.

Or, if you prefer to host the artifacts in a different directory:

```python
attn = LiquidMemory(d_model=512, num_heads=8, artifact_dir="/opt/liquid_memory/dist")
```

### Verifying the install

```python
import torch
from liquid_memory import LiquidMemory

attn = LiquidMemory(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(1, 2048, 512, device="cuda", dtype=torch.float32)
y, _ = attn(x, x, x, is_causal=True)
assert y.shape == x.shape
print("Liquid Memory ready.")
```

On first forward the loader prints the path of the artifact it picked:

```
[liquid_memory_loader] loading dist_public/LiquidMemory_AOTI_sm_90_L2048_trained.pt2
```

If you see `FileNotFoundError: No artifact matching arch sm_XX in dist_public/`,
the shipped artifact set does not include your GPU's compute capability.
Contact the Liquid Memory team for a build, or compile from source with
the `build_matrix.py` tool included in the engineering repository.

---

## 2. Licensing and integrity

### Integrity verification (active today)

`liquid_memory_loader.py` verifies each `.pt2`'s SHA-256 against
`dist_public/MANIFEST.md` before loading it. A tampered or unknown
artifact is refused with a `RuntimeError` and never executed. AOTI
artifacts contain compiled native code, so this check is the runtime
boundary that ensures only artifacts whose hashes you have audited
ever run in your process. The manifest is human-readable; we recommend
pinning your install to a specific repo commit and pre-computing
sha256s out-of-band as defense in depth.

To override (local development only, at your own risk):

```python
from liquid_memory import LiquidMemory
attn = LiquidMemory(d_model, n_heads, verify_artifact=False)
```

### License-key enforcement (roadmap, not active)

The published `liquid_memory_auth.so` registers a
`torch.ops.liquid_memory_auth` namespace as the integration surface
for an Ed25519 signed-launch check. Signature verification is **not
currently enforced**; current artifacts are gated by repository access
only.

The enforcement layer is on the roadmap. When it ships, an
`LM_PRIVATE_KEY` environment variable carrying a 32-byte Ed25519 seed
(provisioned by your Liquid Memory account contact) will be required
to construct a `LiquidMemory(...)` module. Until then, this section
documents the planned interface so customers can budget for it; it is
not a current installation step.

If you require enforced runtime licensing today, email the founders
before integrating - we can fast-track a customer-specific build.

---

## 3. Migration recipes

### 3.1 Vanilla Transformer block

```diff
  import torch.nn as nn
+ from liquid_memory import LiquidMemory

  class Block(nn.Module):
      def __init__(self, d_model, n_heads, dropout=0.1):
          super().__init__()
-         self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
+         self.attn = LiquidMemory(d_model, n_heads, dropout=dropout, batch_first=True)
          self.ln1 = nn.LayerNorm(d_model)
          self.mlp = MLP(d_model)
          self.ln2 = nn.LayerNorm(d_model)

      def forward(self, x):
          h, _ = self.attn(x, x, x, is_causal=True)
          x = self.ln1(x + h)
          x = self.ln2(x + self.mlp(x))
          return x
```

No other changes. Forward shapes, return types, masking semantics
(`is_causal`, `key_padding_mask`) are preserved.

### 3.2 Hugging Face Transformers — patching `LlamaAttention`

```python
from transformers.models.llama.modeling_llama import LlamaAttention
from liquid_memory import LiquidMemory
import torch.nn as nn

class LiquidLlamaAttention(nn.Module):
    """Drop-in for LlamaAttention. Ignores rotary embeddings — the state
    operator is position-aware by construction."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.lm = LiquidMemory(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
            spectral_profile="long_range" if config.max_position_embeddings > 16384 else "balanced",
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
        out, _ = self.lm(hidden_states, hidden_states, hidden_states, is_causal=True)
        return (out, None, past_key_value)

# Patch the model
for layer in model.model.layers:
    layer.self_attn = LiquidLlamaAttention(model.config)
```

> Note: rotary position embeddings (RoPE) become a no-op under Liquid Memory.
> The state operator encodes order through its recurrence, not through
> input rotations. Removing the RoPE call is a minor speedup; leaving it in
> place is harmless.

### 3.3 Autoregressive decode — `O(1)` per-token

```python
from liquid_memory import LiquidMemory

attn = LiquidMemory(d, h).cuda().bfloat16().eval()

# Prefill: parallel mode over the prompt.
prompt_tokens = embed(prompt_ids)               # (B, T_prompt, D)
_ = attn(prompt_tokens, prompt_tokens, prompt_tokens, is_causal=True)
# (For prefill that must also seed the recurrent state, see § 5.)

# Decode: one token at a time, constant memory and latency.
attn.reset_state()
token = embed(start_token_id)                   # (B, D)
for _ in range(max_new):
    out = attn.step(token)                      # (B, D), O(1) in context
    token = embed(sample(out))
```

The hidden state lives on the module and persists across `.step()` calls.
Always call `.reset_state()` before a new generation stream or a batch-size
change.

---

## 4. Long-context training (64k–128k)

Liquid Memory's parallel scan is linear in sequence length, so 128k contexts
fit comfortably on a single H100. Past 64k tokens, the module engages
**precision-safe mode** automatically:

```python
attn = LiquidMemory(4096, 32, batch_first=True).cuda().bfloat16()
x = torch.randn(1, 131_072, 4096, device="cuda", dtype=torch.bfloat16)
y, _ = attn(x, x, x, is_causal=True)   # precision-safe path engaged
```

This is transparent — the input/output dtypes are unchanged; the kernel
promotes interior accumulators only where required. The threshold is fixed
at 64k tokens.

For workloads that live primarily in the long-context regime, pin the
spectral profile explicitly:

```python
attn = LiquidMemory(
    4096, 32,
    spectral_profile="ultra_long",   # 64k–128k regime
    discretization="stable",         # conservative step policy
)
```

| `spectral_profile` | Recommended context range |
| ------------------ | ------------------------- |
| `balanced`         | up to ~16k                |
| `long_range`       | 16k – 64k                 |
| `ultra_long`       | 64k – 128k                |

---

## 5. Benchmarking

A one-shot comparison harness ships with the module:

```python
from liquid_memory import LiquidMemory

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

```
 seq_len │     LM mem │     TX mem │    LM ms │    TX ms │  speedup
─────────────────────────────────────────────────────────────────────
    2048 │     0.21GB │     0.34GB │      1.2 │      2.7 │     2.3x
    8192 │     0.84GB │     5.12GB │      4.7 │     41.3 │     8.8x
   32768 │     3.36GB │       OOM  │     18.6 │     OOM  │      n/a
  131072 │    13.42GB │       OOM  │     74.2 │     OOM  │      n/a
```

Numbers are illustrative; your hardware and dtype mix will vary.

---

## 6. API surface

`LiquidMemory(embed_dim, num_heads, *, dropout=0.0, bias=True, batch_first=True, d_state=64, spectral_profile="balanced", discretization="auto", device=None, dtype=None)`

| Method                | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `forward(q, k, v, …)` | Parallel mode. Drop-in for `nn.MultiheadAttention`. |
| `step(token)`         | Recurrent single-position update, `O(1)`.       |
| `reset_state()`       | Clear the recurrent hidden-state cache.         |
| `benchmark(…)`        | Static. VRAM/latency vs. quadratic baseline.    |

Returns from `forward` are always `(output, None)` — the operator does not
materialize attention weights.

---

## 7. Troubleshooting

| Symptom                                                          | Resolution                                                                                                  |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `No artifact matching arch sm_XX in dist_public/…`               | The shipped artifact set does not include a build for your GPU's compute capability. Contact us for a build, or compile from source. |
| `sha256 mismatch for …` or `No sha256 entry for …`               | The loader refused to load a `.pt2` whose hash is not in `MANIFEST.md`. The artifact has been tampered with or the manifest is stale; do not bypass without auditing. See § 2 (Integrity verification). |
| `Batch size changed from N to M without an intervening reset_state()` | Call `attn.reset_state()` between independent generation streams or batch reconfigurations.                  |
| `Sequence length X exceeds the supported maximum of 131072`      | Out of the supported window. Contact support for extended-context licensing.                                 |

For anything else, capture the stack trace plus the output of
`python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"`
and forward to your account contact.

---

## 8. Frequently asked questions

**Does Liquid Memory require positional embeddings?**
No. The state operator is order-aware by construction. RoPE and absolute
position embeddings are no-ops; you can remove them or leave them in place.

**Can I mix Liquid Memory layers with vanilla attention layers?**
Yes. They share the same I/O shape, so hybrid stacks work without glue
code. A common pattern is alternating layers, or putting Liquid Memory in
the deeper half where long-range mixing matters most.

**What happens if `is_causal=False`?**
The backend runs a bidirectional pass. This costs one extra kernel
invocation; throughput drops by roughly 1.7× versus the causal path.

**Is `torch.compile` supported?**
Yes. The kernel is registered as a custom op and is treated as a black box
by `torch.compile` — surrounding Python/PyTorch is fused normally.

**What precision are the internal accumulators?**
Determined by the kernel and the current execution mode. The external
dtype contract (`bfloat16` in, `bfloat16` out) is preserved. Precision-safe
mode (≥64k tokens) promotes specific interior accumulators only.

**Can I serialize a `LiquidMemory` module?**
Yes — `torch.save(model.state_dict())` works as for any `nn.Module`. The
auth handshake and recurrent-state cache are stripped from pickled state
and re-established on load.

---

© 2026 Liquid Memory, Inc. Distribution and reverse engineering of the
compiled kernel are governed by your license agreement.

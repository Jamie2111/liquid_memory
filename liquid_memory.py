# =====================================================================
# liquid_memory.py - drop-in MultiheadAttention shim backed by a Mode 2
# AOTI artifact.
# ---------------------------------------------------------------------
# This is the integration surface advertised by README_INTEGRATION.md:
#
#     from liquid_memory import LiquidMemory
#     self.attn = LiquidMemory(embed_dim=d_model, num_heads=n_heads,
#                              batch_first=True)
#     out, _ = self.attn(x, x, x, is_causal=True)
#
# Internally it:
#   - lazy-loads a Mode 2 AOTI artifact via liquid_memory_loader.load(),
#     picking the .pt2 whose embedded arch tag matches the host GPU and
#     whose seq_len tag matches the input length;
#   - caches loaded artifacts by (seq_len, device) so subsequent calls
#     at the same shape skip reload overhead;
#   - returns (output, None) to match the (output, attn_weights) tuple
#     that nn.MultiheadAttention returns. SSMs do not expose attention
#     weights; the second slot is always None.
#
# Limitations vs nn.MultiheadAttention:
#   - SELF-ATTENTION ONLY. query must equal key and value (the SSM is a
#     sequence-mixing operator with no separate Q/K/V projections).
#   - is_causal=True is the only supported masking mode. The SSM scan
#     is intrinsically causal; passing is_causal=False does not change
#     that.
#   - No attn_mask, no key_padding_mask, no need_weights output.
#   - Fixed-shape AOTI: each compiled artifact handles one specific
#     seq_len. To support multiple sequence lengths, build artifacts at
#     each seq_len via build_matrix.py and the loader picks the right
#     one per input.
# =====================================================================

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from liquid_memory_loader import load as _load_artifact, current_arch


class LiquidMemory(nn.Module):
    """Drop-in replacement for torch.nn.MultiheadAttention powered by a
    Mode 2 AOTI artifact compiled for the host GPU.

    Constructor signature matches nn.MultiheadAttention closely. Args
    the SSM does not use (dropout, vdim, kdim, ...) are accepted and
    ignored for source-compatibility.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device=None,
        dtype=None,
        *,
        artifact_dir: str = "dist_public",
        **_unused,
    ):
        super().__init__()
        if not batch_first:
            raise NotImplementedError(
                "LiquidMemory only supports batch_first=True. nn.MultiheadAttention's "
                "batch_first=False is convertible by .transpose(0, 1) at the call site."
            )
        if dropout != 0.0:
            print(f"[LiquidMemory] note: dropout={dropout} ignored (SSM has no "
                  f"attention weights to drop).")
        if add_bias_kv or add_zero_attn:
            print(f"[LiquidMemory] note: add_bias_kv / add_zero_attn ignored "
                  f"(not applicable to an SSM).")
        if (kdim is not None and kdim != embed_dim) or (vdim is not None and vdim != embed_dim):
            raise NotImplementedError(
                "LiquidMemory does not support distinct kdim/vdim - the SSM has no "
                "Q/K/V projections."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.artifact_dir = artifact_dir

        # (seq_len, device_name) -> loaded AOTICompiledModel
        self._artifact_cache: dict = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _device_key(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    def _get_artifact(self, seq_len: int):
        key = (seq_len, self._device_key())
        cached = self._artifact_cache.get(key)
        if cached is not None:
            return cached
        artifact = _load_artifact(
            search_dir=self.artifact_dir,
            seq_len=seq_len,
        )
        self._artifact_cache[key] = artifact
        return artifact

    # ------------------------------------------------------------------
    # Forward (matches nn.MultiheadAttention call signature)
    # ------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention: default missing args to query.
        if key is None:
            key = query
        if value is None:
            value = query

        # Validate self-attention assumption. Skip the full elementwise
        # equality check when key/value are the same tensor object as
        # query (the common case after the defaulting above).
        if key is not query:
            if key.shape != query.shape or not torch.equal(key, query):
                raise NotImplementedError(
                    "LiquidMemory is a self-attention replacement: query, key, "
                    "and value must be the same tensor (or 'key' and 'value' "
                    "must be omitted). Got distinct key tensor."
                )
        if value is not query:
            if value.shape != query.shape or not torch.equal(value, query):
                raise NotImplementedError(
                    "LiquidMemory is a self-attention replacement: query, key, "
                    "and value must be the same tensor. Got distinct value tensor."
                )

        if not is_causal:
            # The SSM is causal by construction; we cannot make it acausal.
            # Warn but do not error: many callers pass is_causal=False as a
            # default and then add a separate attn_mask, which we ignore.
            pass
        if attn_mask is not None:
            print("[LiquidMemory] note: attn_mask is ignored.")
        if key_padding_mask is not None:
            print("[LiquidMemory] note: key_padding_mask is ignored.")

        x = query
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, L, embed_dim) for batch_first=True; "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Input embed_dim={x.shape[-1]} does not match the "
                f"LiquidMemory layer's embed_dim={self.embed_dim}."
            )

        seq_len = x.shape[1]

        # AOTI artifacts are compiled for fp32 inputs. Cast in, cast back out.
        orig_dtype = x.dtype
        if x.dtype != torch.float32:
            x_fp32 = x.to(torch.float32)
        else:
            x_fp32 = x
        if not x_fp32.is_cuda:
            raise RuntimeError(
                "LiquidMemory AOTI artifacts run on CUDA only; move input to GPU."
            )
        x_fp32 = x_fp32.contiguous()

        artifact = self._get_artifact(seq_len)
        with torch.inference_mode():
            y = artifact(x_fp32)

        if y.dtype != orig_dtype:
            y = y.to(orig_dtype)
        # MHA returns (output, attention_weights). We have no weights.
        return y, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"batch_first={self.batch_first}, "
            f"artifact_dir={self.artifact_dir!r} "
            f"[host arch: {current_arch() if torch.cuda.is_available() else 'cpu'}]"
        )


# ---------------------------------------------------------------------
# CLI: quick smoke test. `python liquid_memory.py` builds a LiquidMemory
# layer and runs a forward against whatever artifact matches the host.
# ---------------------------------------------------------------------
def _cli() -> int:
    import sys

    if not torch.cuda.is_available():
        print("CUDA required for LiquidMemory.")
        return 1

    try:
        seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    except ValueError:
        print(f"usage: {sys.argv[0]} [seq_len]")
        return 1

    print(f"[smoke] building LiquidMemory(embed_dim=512, num_heads=8)")
    attn = LiquidMemory(embed_dim=512, num_heads=8, batch_first=True)
    print(f"[smoke] {attn}")

    x = torch.randn(1, seq_len, 512, device="cuda", dtype=torch.float32)
    print(f"[smoke] forward at seq_len={seq_len}")
    out, weights = attn(x, x, x, is_causal=True)
    print(f"[smoke] output shape: {tuple(out.shape)}, finite: "
          f"{bool(torch.isfinite(out).all())}")
    print(f"[smoke] attention weights returned: {weights}  (None is correct - SSMs have none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

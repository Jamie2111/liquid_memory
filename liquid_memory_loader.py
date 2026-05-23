# =====================================================================
# liquid_memory_loader.py - runtime AOTI artifact selector.
# ---------------------------------------------------------------------
# Customers and internal benchmarks call load() to obtain a callable
# Mode 2 module that runs the .pt2 matching the current GPU's compute
# capability. The selector picks among artifacts named
# LiquidMemory_AOTI_<arch>[_<tag>].pt2 produced by build_matrix.py.
#
# Selection order:
#   1. Exact arch match (e.g. sm_90 on an H100). If multiple tags exist
#      for that arch and one matches the requested seq_len, prefer it.
#   2. If no arch match exists, raise FileNotFoundError with a clear
#      message that build_matrix.py needs to be run on this GPU.
#
# Usage:
#     from liquid_memory_loader import load
#     attn = load(seq_len=2048)            # picks the right .pt2
#     y = attn(torch.randn(1, 2048, 512, device="cuda", dtype=torch.float32))
# =====================================================================

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch


def current_arch() -> str:
    """Return e.g. 'sm_90' for the GPU at index 0."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    p = torch.cuda.get_device_properties(0)
    return f"sm_{p.major}{p.minor}"


def _parse_artifact_name(path: Path) -> Optional[Tuple[str, Optional[str]]]:
    """Parse LiquidMemory_AOTI_<arch>[_<tag>].pt2 into (arch, tag). Returns
    None if the name does not match. Tag may be None for the legacy form
    without a tag."""
    stem = path.stem
    # Tagged form: LiquidMemory_AOTI_sm_90_L2048
    m = re.match(r"^LiquidMemory_AOTI_(sm_\d+[a-z]?)(?:_(.+))?$", stem)
    if not m:
        return None
    return (m.group(1), m.group(2))


def find_artifact(
    search_dir: str = "dist_public",
    arch: Optional[str] = None,
    seq_len: Optional[int] = None,
) -> Path:
    """Find the best-matching .pt2 in search_dir for the given (arch, seq_len).

    arch defaults to the current GPU arch. seq_len, if given, prefers an
    artifact tagged 'L<seq_len>'.

    Selection is exact-arch only. Cross-arch fallbacks are not allowed:
    AOTI cubins are compiled per arch and will fail at runtime with
    'no kernel image is available' if mismatched (we measured this).
    """
    arch = arch or current_arch()
    search = Path(search_dir)
    if not search.exists():
        raise FileNotFoundError(
            f"Search directory not found: {search.resolve()}. "
            f"Did you forget to checkout dist_public/ ?"
        )

    candidates = []
    for path in sorted(search.glob("*.pt2")):
        parsed = _parse_artifact_name(path)
        if parsed is None:
            continue
        parsed_arch, parsed_tag = parsed
        if parsed_arch != arch:
            continue
        candidates.append((path, parsed_tag))

    if not candidates:
        # Maybe there is a tag-less LiquidMemory_AOTI.pt2 left over from
        # the old build flow. Surface a helpful error rather than
        # silently using it - using an artifact whose arch we cannot
        # verify is exactly how this whole thing broke.
        raise FileNotFoundError(
            f"No artifact matching arch {arch} in {search.resolve()}.\n"
            f"Run  python build_matrix.py  on a {arch} GPU to produce one,\n"
            f"then commit the resulting LiquidMemory_AOTI_{arch}_<tag>.pt2 file."
        )

    if seq_len is not None:
        target_tag = f"L{seq_len}"
        for p, tag in candidates:
            if tag == target_tag:
                return p
        # Fall through to first candidate with a warning.
        print(f"[liquid_memory_loader] no artifact tagged {target_tag} for arch "
              f"{arch}; using {candidates[0][0].name} (tag {candidates[0][1]!r}).")

    return candidates[0][0]


def load(
    search_dir: str = "dist_public",
    arch: Optional[str] = None,
    seq_len: Optional[int] = None,
):
    """Load the AOTI artifact for the current GPU.

    Returns an AOTICompiledModel that you call with a single input
    tensor of shape (B, L, d_model), fp32, on CUDA.
    """
    path = find_artifact(search_dir=search_dir, arch=arch, seq_len=seq_len)
    print(f"[liquid_memory_loader] loading {path}")
    loaded = torch._inductor.aoti_load_package(str(path))
    return loaded


# -----------------------------------------------------------------------
# CLI: `python liquid_memory_loader.py` lists the artifacts on disk and
# checks which one matches the current GPU.
# -----------------------------------------------------------------------
def _cli() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    arch = current_arch()
    name = torch.cuda.get_device_name(0)
    print(f"current GPU: {name} ({arch})")
    here = Path(__file__).resolve().parent
    search = here / "dist_public"
    if not search.exists():
        print(f"dist_public not found at {search}")
        return 1
    print(f"\nartifacts in {search}:")
    any_match = False
    for path in sorted(search.glob("*.pt2")):
        parsed = _parse_artifact_name(path)
        if parsed is None:
            print(f"  {path.name}  (unrecognized name)")
            continue
        parsed_arch, parsed_tag = parsed
        marker = " <-- matches current GPU" if parsed_arch == arch else ""
        if parsed_arch == arch:
            any_match = True
        print(f"  {path.name}  arch={parsed_arch} tag={parsed_tag}{marker}")
    if not any_match:
        print(f"\nNo artifact matches {arch}. Run  python build_matrix.py  to build one.")
        return 1
    print()
    try:
        path = find_artifact(search_dir=str(search))
        print(f"selected: {path}")
        loaded = torch._inductor.aoti_load_package(str(path))
        print(f"loaded type: {type(loaded).__name__}")
        return 0
    except Exception as e:
        print(f"load FAILED: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())

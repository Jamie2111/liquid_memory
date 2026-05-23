# LiquidMemory AOT Distribution Manifest

compile_dtype=float32
expected_input_dtype=float32

## Compiled artifacts (per GPU compute capability x sequence length)

Each `.pt2` is compiled for a specific compute capability via PyTorch's
AOT-Inductor backend and is functional only on a matching GPU. The
runtime loader (`liquid_memory_loader.py`) selects the artifact whose
filename matches the host GPU's compute capability and the input's
sequence length.

### sm_89 (NVIDIA RTX 40-series, L40, L40S)

LiquidMemory_AOTI_sm_89_L2048_trained.pt2  sha256=b7e13b6fc3d99f8eb1e30596fc9f956bc4210f5162027068bef0e8816f3a57e9

### sm_90 (NVIDIA H100, H200)

LiquidMemory_AOTI_sm_90_L2048_trained.pt2   sha256=f3704b872e658db3868074ca105b769d2c4b550a5e62eb517de531790e1da1ff
LiquidMemory_AOTI_sm_90_L4096_trained.pt2   sha256=a9516f0df463dcc7d5f590993717c0fe2f03f1e22995c3f6ae2ab223f817088f
LiquidMemory_AOTI_sm_90_L8192_trained.pt2   sha256=507217f49f27fcd50e49fdb6ed24c3a1b1c6dafece3b9d4e78f76c9cef7a88ea
LiquidMemory_AOTI_sm_90_L16384_trained.pt2  sha256=4c0394a736e737f2c854043ea22269ba332dc99f18af8ddf3397c95a956ae99e
LiquidMemory_AOTI_sm_90_L32768_trained.pt2  sha256=26b9e1b78dc399bf5f1162ad709d0367c3b5f7d5246e7587a4c11da89fae6c02
LiquidMemory_AOTI_sm_90_L65536_trained.pt2  sha256=c1f711773873ded2182db7622afd0fc1ab124bc420b7f01bf9b59b5e9cb4b56c

## Auth library (placeholder)

liquid_memory_auth.so  sha256=eb7ff171fb7e13f87c9b163a4b1ef72fd43316078ac4af1efb58ba2869b54b0f

Note: the current `liquid_memory_auth.so` registers the
`torch.ops.liquid_memory_auth` namespace but does not enforce
signature verification. License-gating is on the roadmap and is not
active in this distribution; current artifacts are gated by repo
access only.

## Block configuration

Every artifact above was AOT-compiled from a single
`LiquidMemoryBlock(d_model=512, d_state=16, d_conv=4, expand=2,
scan_chunk_size=16)` with trained weights from a TinyStories-trained
6-layer character-level LM (val perplexity 2.17 at 3,000 steps,
batch 4, seq_len 512). See AOTI_METADATA.json for the per-artifact
input shape contract.

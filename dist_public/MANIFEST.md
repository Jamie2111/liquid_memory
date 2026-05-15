# LiquidMemory AOT Graduation Manifest

compile_dtype=float32
expected_input_dtype=float32

LiquidMemory_AOTI.pt2  sha256=181503bc20ae710ad9119d4c26b8c310f75bc5157a07c58d61823c81874a3b4c
liquid_memory_auth.so  sha256=fab67377e2a5d31ebe507cf119d651972a86ac67f02c76a8713c0674977e2381

Runtime launch token format:
  LIQUID_MEMORY_AOT_CHALLENGE='LMv1|exp=<unix_seconds>|nonce=<random>|scope=<deployment>'
  LIQUID_MEMORY_AOT_SIGNATURE_HEX='<ed25519 signature over the challenge bytes>'

Runtime metadata: AOTI_METADATA.json
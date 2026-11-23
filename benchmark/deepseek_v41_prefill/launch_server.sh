#!/usr/bin/env bash
# Run inside the reproduction container; no inherited PYTHONPATH overlays.
set -euo pipefail
ulimit -c 0
export PATH=/opt/sglang/bin:$PATH
export PYTHONPATH=/repro/deps:/repro/source/python
export SGLANG_CACHE_DIR=/repro/cache
export SGLANG_OPT_DSV41_SM90=1
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}'
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 /repro/tools/manifest.py runtime /repro/results
command=(python3 -m sglang.launch_server
  --model-path /models/DeepSeek-V4.1-Flash-1017
  --served-model-name DeepSeek-V4.1-Flash
  --host 0.0.0.0 --port "${PREFILL_PORT:-30002}" --admin-api-key profile
  --trust-remote-code --language-model-only --model-impl sglang
  --tp-size 8 --enable-prefill-cp --cp-strategy interleave
  --enable-cp-decode-attn-tp --moe-runner-backend humming
  --enable-decoder-swa-bounded-replay --swa-full-tokens-ratio 0.05
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --mem-fraction-static 0.84 --max-running-requests 64
  --cuda-graph-max-bs-decode 64 --chunked-prefill-size 16384
  --disable-radix-cache)
printf '%q ' "${command[@]}" > /repro/results/launch_command.txt
printf '\n' >> /repro/results/launch_command.txt
exec "${command[@]}"

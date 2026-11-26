#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export PATH=/opt/sglang/bin:$PATH
export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1
export PYTHONPATH=/repro/deps:/repro/source/python
export SGLANG_CACHE_DIR=/repro/cache
export SGLANG_OPT_DSV41_SM90=1
export SGLANG_OPT_DSV41_SM90_LOSSY=fp8_lm_head_target
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
export SGLANG_EMBEDDING_POOL_SIZE_MB=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RAGGED_VERIFY_MODE=static
if [[ "${DECODE_MODE:-decode}" == accuracy ]]; then
  unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_GREEDY SGLANG_DSPARK_SIMULATE_FOLD
  role_args=(--disable-radix-cache)
else
  export SGLANG_SIMULATE_ACC_LEN=4.5
  role_args=(--disaggregation-mode decode)
fi
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=
python3 /repro/tools/manifest.py runtime /repro/results
parallel_args=(--tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head)
if [[ "${DECODE_IMPLEMENTATION:-ant}" == local ]]; then
  parallel_args=(--tp-size 8 --dp-size 1 --attn-dp-size 8 --enable-dp-lm-head)
fi
command=(python3 -m sglang.launch_server
  --model-path /models/DeepSeek-V4.1-Flash-1017
  --random-seed 42 --trust-remote-code --host 0.0.0.0 --port 8089 --admin-api-key profile
  "${parallel_args[@]}"
  --ep-size "${DECODE_EP:-1}"
  --moe-runner-backend humming --swa-full-tokens-ratio 0.05
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --mem-fraction-static 0.85 --max-running-requests 256 --cuda-graph-max-bs-decode 32
  --chunked-prefill-size 16384 --tokenizer-worker-num 8 --load-balance-method total_tokens
  "${role_args[@]}" --language-only --watchdog-timeout 1800
  --enable-cache-report --reasoning-parser auto --tool-call-parser auto)
printf '%q ' "${command[@]}" > /repro/results/launch_command.txt
printf '\n' >> /repro/results/launch_command.txt
exec "${command[@]}"

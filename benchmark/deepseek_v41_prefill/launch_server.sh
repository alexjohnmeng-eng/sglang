#!/usr/bin/env bash
# Run inside the reproduction container; no inherited PYTHONPATH overlays.
set -euo pipefail
ulimit -c 0
export PATH=/opt/sglang/bin:$PATH
export PYTHONPATH=/repro/deps:/repro/source/python
export SGLANG_JIT_CACHE_DIR=/repro/cache/sgl_kernel_jit
if [[ -d /repro/flashinfer ]]; then export PYTHONPATH="/repro/flashinfer:$PYTHONPATH"; fi
if [[ -d /repro/dense_deepgemm ]]; then
  export PYTHONPATH="/repro/dense_deepgemm:$PYTHONPATH"
  export DG_JIT_CACHE_DIR=/repro/cache/deepgemm-wqb-q8-rope
fi
if [[ "${SGLANG_EXPERIMENT_DSV41_SM90_MEGAMOE_DEEPGEMM:-0}" == 1 ]]; then
  export PYTHONPATH="/repro/deepgemm473:$PYTHONPATH"
  export DG473_JIT_CACHE_DIR=/repro/cache/deepgemm473
fi
if [[ -d /repro/deepgemm521 ]]; then
  export PYTHONPATH="/repro/deepgemm521:$PYTHONPATH"
  export DG521_JIT_CACHE_DIR=/repro/cache/deepgemm521
fi
export SGLANG_CACHE_DIR=/repro/cache
export SGLANG_OPT_DSV41_SM90=1
if [[ "${PREFILL_EP8_HISTORICAL:-0}" == 1 ]]; then
  [[ "${PREFILL_EP_SIZE:-1}" == 8 ]] || { echo "Historical prefill requires EP8" >&2; exit 1; }
  export SGLANG_OPT_DSV41_CP8_EP8_PREFILL=1
  export SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN=1
  export SGLANG_OPT_DSV41_FLASHINFER_SM90_MAX_TOKENS=8192
else
  unset SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN
  unset SGLANG_OPT_DSV41_FLASHINFER_SM90_MAX_TOKENS
fi
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}'
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV41_HUMMING_PREFILL_NTAIL="${PREFILL_HUMMING_NTAIL:-0}"

python3 /repro/tools/manifest.py runtime /repro/results
command=(python3 -m sglang.launch_server
  --model-path /models/DeepSeek-V4.1-Flash-1017
  --served-model-name DeepSeek-V4.1-Flash
  --host 0.0.0.0 --port "${PREFILL_PORT:-30002}" --admin-api-key profile
  --trust-remote-code --language-model-only --model-impl sglang
  --tp-size 8 --enable-prefill-cp --cp-strategy interleave
  --ep-size "${PREFILL_EP_SIZE:-1}"
  --enable-cp-decode-attn-tp --moe-runner-backend humming
  --enable-decoder-swa-bounded-replay --swa-full-tokens-ratio 0.05
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --mem-fraction-static "${PREFILL_MEM_FRACTION:-0.84}" --max-running-requests 64
  --cuda-graph-max-bs-decode 64 --chunked-prefill-size "${PREFILL_CHUNK_SIZE:-16384}"
  --max-prefill-tokens "${PREFILL_CHUNK_SIZE:-16384}"
  --disable-radix-cache)
if [[ "${PREFILL_EP8_HISTORICAL:-0}" == 1 ]]; then
  command+=(--ep-num-redundant-experts 64 --ep-dispatch-algorithm dynamic --init-expert-location /repro/ep-placement.json)
fi
if [[ "${PREFILL_ATTENTION_BACKEND:-auto}" != auto ]]; then
  command+=(--dsv4-prefill-backend "$PREFILL_ATTENTION_BACKEND")
fi
printf '%q ' "${command[@]}" > /repro/results/launch_command.txt
printf '\n' >> /repro/results/launch_command.txt
exec "${command[@]}"

#!/usr/bin/env bash
# Compare cache-off CP8/EP8 chunk sizes and run full GSM8K on the same server.
set -euo pipefail

tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$tools" rev-parse --show-toplevel)
runtime=${PREFILL_RUNTIME_DIR:-/data1/fjmeng2/dsv41-runtime}
previous=$runtime/repro/prefill-110k-20261124
experiments=$runtime/repro/prefill-110k-local-20261125
dense=$experiments/wqb-q8-rope
chunk=${1:-262144}
case "$chunk" in
  65536) export PREFILL_DEEPGEMM473_16K=0 ;;
  131072|262144) export PREFILL_DEEPGEMM473_16K=1 ;;
  *) echo "Usage: $0 [65536|131072|262144]" >&2; exit 2 ;;
esac

if [[ ${PREFILL_DENSE_MULTICAST:-0} == 1 ]]; then
  dense=$experiments/dense-multicast
  export PREFILL_DEEPGEMM521_SITE="${PREFILL_DEEPGEMM521_SITE:-$experiments/mega-mhc-521/site}"
fi

run_id=$(date -u +%Y%m%dT%H%M%SZ)
export PREFILL_OUTPUT="${PREFILL_OUTPUT:-$runtime/repro/cp8-megamoe-$chunk-$run_id}"
export PREFILL_CONTAINER="${PREFILL_CONTAINER:-dsv41-cp8-$chunk-$run_id}"
export PREFILL_CHUNK_SIZE="$chunk" PREFILL_ATTENTION_BACKEND=flashmla_sparse_q8
export PREFILL_EP_SIZE=8 PREFILL_MEM_FRACTION=0.74
export PREFILL_EP8_HISTORICAL=1
export PREFILL_HUMMING_NTAIL=1 PREFILL_SHARED_TP1=1 PREFILL_WQB_Q8_ROPE=1
export PREFILL_Q8_TMA_SWA="${PREFILL_Q8_TMA_SWA:-1}"
export PREFILL_FLASHINFER_SITE="${PREFILL_FLASHINFER_SITE:-$runtime/fi075/site}"
export PREFILL_EP_PLACEMENT="${PREFILL_EP_PLACEMENT:-$previous/routes-calibration-v1/placement-variable_copies.json}"
export PREFILL_DEEPGEMM473_SITE="${PREFILL_DEEPGEMM473_SITE:-$previous/upstream-20261125/deepgemm-473-prefill-tuning-site}"
export PREFILL_DENSE_DEEPGEMM_SITE="${PREFILL_DENSE_DEEPGEMM_SITE:-$dense/site}"
export PREFILL_LONG_PROBE="${PREFILL_LONG_PROBE:-$previous/probe_long_prefill.py}"
exec bash "$runtime/repro/prefill-plus20-20261124/run_prefill_with_gsm8k.sh" "$repo"

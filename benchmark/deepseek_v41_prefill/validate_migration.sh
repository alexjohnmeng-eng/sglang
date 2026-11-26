#!/usr/bin/env bash
# Host entrypoint: immutable-source DP8/EP8, CP8/EP8 and EP1 validation.
set -euo pipefail

tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$tools" rev-parse --show-toplevel)
output=${1:?Usage: validate_migration.sh NEW_OUTPUT_ROOT [REVISION_OR_TREE]}
revision=$(git -C "$repo" rev-parse --verify "${2:-HEAD}^{tree}")
runtime=${PREFILL_RUNTIME_DIR:-/data1/fjmeng2/dsv41-runtime}
if [[ -n "${PREFILL_UNIFIED_DEEPGEMM_SITE:-}" ]]; then
  exec bash "$tools/dependencies/unified/validate.sh" \
    "$PREFILL_UNIFIED_DEEPGEMM_SITE" "$output" "$revision"
fi
# Legacy dependency sites below are retained for historical comparisons only.
mhc=${PREFILL_DEEPGEMM521_SITE:-$runtime/repro/prefill-110k-local-20261125/mega-mhc-521/site}
mkdir -- "$output"
output=$(realpath "$output")
printf '%s\n' "$revision" > "$output/source-tree.txt"
cd "$repo"

python3 "$tools/decode_regression/run.py" \
  --revision "$revision" --baseline "${DECODE_BASELINE_REVISION:-82bebb1d34}" \
  --runtime "$runtime" --cases accuracy control candidate \
  --output "$output/dp8ep8" 2>&1 | tee "$output/dp8ep8-driver.log"

# The historical path uses the image FlashInfer and pinned DeepGEMM sites.
env -u PREFILL_FLASHINFER_SITE \
  PREFILL_REVISION="$revision" PREFILL_DENSE_MULTICAST=1 \
  PREFILL_DEEPGEMM521_SITE="$mhc" \
  PREFILL_OUTPUT="$output/cp8ep8" \
  bash "$tools/run_cp8_megamoe.sh" 262144 2>&1 | tee "$output/cp8ep8-driver.log"

# EP1 retains paired W2, stock dense GEMM and its existing Mega mHC site.
env -u PREFILL_FLASHINFER_SITE -u PREFILL_DENSE_DEEPGEMM_SITE \
  -u PREFILL_DEEPGEMM473_SITE -u PREFILL_EP_PLACEMENT \
  PREFILL_REVISION="$revision" PREFILL_EP8_HISTORICAL=0 \
  PREFILL_EP_SIZE=1 PREFILL_CHUNK_SIZE=65536 PREFILL_MEM_FRACTION=0.84 \
  PREFILL_ATTENTION_BACKEND=flashmla_sparse_q8 PREFILL_HUMMING_NTAIL=1 \
  PREFILL_SHARED_TP1=1 PREFILL_W2_PAIR_LAYERS=40 PREFILL_RUN_GSM8K=1 \
  PREFILL_DENSE_MULTICAST=0 PREFILL_Q8_TMA_SWA=0 \
  PREFILL_DEEPGEMM521_SITE="$mhc" PREFILL_OUTPUT="$output/ep1" \
  bash "$tools/run.sh" 2>&1 | tee "$output/ep1-driver.log"

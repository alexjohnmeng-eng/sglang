#!/usr/bin/env bash
# Three reviewed H20 recipes using one DeepGEMM source/wheel. Run on the host.
set -euo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
  echo "Usage: $0 {cp8ep8|cp8ep1|dp8ep8} [--prepare-only]" >&2
  echo "Set PREFILL_UNIFIED_DEEPGEMM_SITE and optionally PREFILL_REVISION/PREFILL_OUTPUT." >&2
  echo "dp8ep8 also accepts decode_regression/run.py options, e.g. --cases accuracy." >&2
}
topology=${1:---help}
case "$topology" in
  cp8ep8|cp8ep1|dp8ep8) shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac
runtime=${PREFILL_RUNTIME_DIR:-/data1/fjmeng2/dsv41-runtime}
export PREFILL_RUNTIME_DIR="$runtime"
export H20_TOPOLOGY="$topology"
export PREFILL_UNIFIED_DEEPGEMM_SITE="${PREFILL_UNIFIED_DEEPGEMM_SITE:?Set the unified DeepGEMM site; see H20_TESTING.md}"
for legacy in PREFILL_DENSE_DEEPGEMM_SITE PREFILL_DEEPGEMM473_SITE PREFILL_DEEPGEMM521_SITE PREFILL_FLASHINFER_SITE; do
  [[ -z "${!legacy:-}" ]] || { echo "Unified H20 recipe conflicts with $legacy" >&2; exit 2; }
done
export PREFILL_UNIFIED_DEEPGEMM_SITE=$(realpath "$PREFILL_UNIFIED_DEEPGEMM_SITE")
python3 "$tools/dependencies/unified/verify.py" site "$PREFILL_UNIFIED_DEEPGEMM_SITE"
export PREFILL_OUTPUT="${PREFILL_OUTPUT:-$runtime/repro/h20-$topology-$(date -u +%Y%m%dT%H%M%SZ)}"

# Fixed recipes: do not inherit a previous experiment's WQB/EP/tile choices.
export PREFILL_WQB_Q8_ROPE=0
export PREFILL_EP8_HISTORICAL=0 PREFILL_EP_SIZE=1
export PREFILL_DENSE_MULTICAST=0 PREFILL_Q8_TMA_SWA=0 PREFILL_DEEPGEMM473_16K=0
export PREFILL_W2_PAIR_LAYERS=0
export PREFILL_ATTENTION_BACKEND=flashmla_sparse_q8
export PREFILL_HUMMING_NTAIL=1 PREFILL_SHARED_TP1=1 PREFILL_RUN_GSM8K=1

case "$topology" in
  cp8ep8)
    export PREFILL_EP8_HISTORICAL=1 PREFILL_EP_SIZE=8
    export PREFILL_CHUNK_SIZE=262144 PREFILL_MEM_FRACTION=0.74
    export PREFILL_DENSE_MULTICAST=1 PREFILL_Q8_TMA_SWA=1 PREFILL_DEEPGEMM473_16K=1
    export PREFILL_EP_PLACEMENT="$tools/dependencies/placement-variable-copies.json"
    exec bash "$tools/run.sh" "$@"
    ;;
  cp8ep1)
    unset PREFILL_EP_PLACEMENT
    export PREFILL_CHUNK_SIZE=65536 PREFILL_MEM_FRACTION=0.84
    export PREFILL_W2_PAIR_LAYERS=40
    exec bash "$tools/run.sh" "$@"
    ;;
  dp8ep8)
    exec python3 "$tools/decode_regression/run.py" \
      --revision "${PREFILL_REVISION:-HEAD}" \
      --baseline "${DECODE_BASELINE_REVISION:-b1ae8c062ddfd809ab5d346795caf8f41aefdaf6}" \
      --runtime "$runtime" --unified-deepgemm-site "$PREFILL_UNIFIED_DEEPGEMM_SITE" \
      --output "$PREFILL_OUTPUT" --cases candidate accuracy "$@"
    ;;
esac

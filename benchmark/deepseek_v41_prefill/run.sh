#!/usr/bin/env bash
# Host entry point. A dedicated container owns the server and all its children.
set -euo pipefail
ulimit -c 0

tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$tools_dir" rev-parse --show-toplevel)
source_revision=$(git -C "$repo_dir" rev-parse --verify "${PREFILL_REVISION:-HEAD}^{commit}")
expected_image=sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152
image=${PREFILL_IMAGE:-lmsysorg/sglang:latest}
model=${PREFILL_MODEL_DIR:-/data1/fjmeng2/models/DeepSeek-V4.1-Flash-1017}
output=${PREFILL_OUTPUT:-/data1/fjmeng2/dsv41-runtime/repro/$(date -u +%Y%m%dT%H%M%SZ)}
deps=${PREFILL_DEPS_DIR:-/data1/fjmeng2/dsv41-runtime/fa3-pylibs}
cache=${PREFILL_CACHE_DIR:-/data1/fjmeng2/dsv41-runtime/sglang-cache-migrated}
port=${PREFILL_PORT:-30002}
container=${PREFILL_CONTAINER:-dsv41-prefill-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${1:-run}
if [[ "$mode" != run && "$mode" != --prepare-only ]]; then
  echo "Usage: $0 [run|--prepare-only]; configuration: see README.md" >&2
  exit 2
fi
# Never reuse an experiment directory or alter the serving checkout.
mkdir -p "$(dirname -- "$output")"
mkdir -- "$output"
output=$(realpath "$output")
mkdir "$output/source" "$output/tools"
cp -a "$tools_dir/." "$output/tools/"
git -C "$repo_dir" archive "$source_revision" > "$output/source.tar"
tar -xf "$output/source.tar" -C "$output/source"
rm "$output/source.tar"
python3 "$output/tools/manifest.py" source "$output" "$source_revision"
git -C "$repo_dir" rev-parse HEAD > "$output/tools_revision.txt"
git -C "$repo_dir" status --short > "$output/checkout_status.txt"
if [[ "$mode" == --prepare-only ]]; then
  echo "Prepared isolated source: $output/source"
  exit 0
fi
[[ -f "$model/config.json" ]] || { echo "Model not found: $model" >&2; exit 1; }
mkdir -p "$deps" "$cache"
model=$(realpath "$model")
deps=$(realpath "$deps")
cache=$(realpath "$cache")
docker image inspect "$image" > "$output/image.json"
image_id=$(docker image inspect "$image" --format '{{.Id}}')
if [[ "$image_id" != "$expected_image" && "${PREFILL_ALLOW_IMAGE_MISMATCH:-0}" != 1 ]]; then
  echo "Image changed: $image_id; expected $expected_image." >&2
  echo "Use the recorded image, or explicitly set PREFILL_ALLOW_IMAGE_MISMATCH=1 for a new experiment." >&2
  exit 1
fi
# Refuse to contend with an existing service; never kill someone else's process.
python3 - "$port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", int(sys.argv[1])))
PY
if [[ ! -d "$deps/sgl_kernel" || ! -d "$deps/sgl_fa3" ]]; then
  docker run --rm --network host -v "$deps:/repro/deps" "$image_id" \
    /opt/sglang/bin/python3 -m pip install --no-deps --target /repro/deps \
    sglang-kernel==0.4.11 sgl-fa3==0.1.0 2>&1 | tee "$output/install.log"
fi
docker ps --format '{{.Names}}' > "$output/existing_containers.txt"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPUs have active compute processes. Stop the previous experiment before running." >&2
  exit 1
fi
owned_container=0
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ "$owned_container" == 1 ]]; then
    docker logs "$container" > "$output/server.log" 2>&1 || true
    if [[ "${PREFILL_KEEP_SERVER:-0}" != 1 ]]; then
      docker rm -f "$container" >/dev/null || true
    fi
  fi
  echo "Artifacts: $output (exit $status)"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
device_args=()
if [[ -d /dev/infiniband ]]; then
  device_args+=(--device /dev/infiniband:/dev/infiniband)
fi
experiment_args=()
if [[ -n "${PREFILL_DEEPGEMM473_SITE:-}" ]]; then
  experiment_args+=(
    -e SGLANG_EXPERIMENT_DSV41_SM90_MEGAMOE_DEEPGEMM=1
    -e "SGLANG_OPT_DSV41_SM90_MEGAMOE_16K_PREFILL=${PREFILL_DEEPGEMM473_16K:-0}"
    -v "$PREFILL_DEEPGEMM473_SITE:/repro/deepgemm473:ro"
  )
fi
if [[ -n "${PREFILL_DEEPGEMM521_SITE:-}" ]]; then
  experiment_args+=(
    -e SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1
    -v "$PREFILL_DEEPGEMM521_SITE:/repro/deepgemm521:ro"
  )
fi
if [[ -n "${PREFILL_DENSE_DEEPGEMM_SITE:-}" ]]; then
  experiment_args+=(
    -v "$PREFILL_DENSE_DEEPGEMM_SITE:/repro/dense_deepgemm:ro"
    -e "SGLANG_EXPERIMENT_DSV41_SM90_WQB_Q8_ROPE=${PREFILL_WQB_Q8_ROPE:-0}"
  )
fi
docker run -d --name "$container" --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
  --cap-add IPC_LOCK "${device_args[@]}" "${experiment_args[@]}" \
  -e PREFILL_PORT="$port" \
  -e PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-16384}" \
  -e PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-auto}" \
  -e PREFILL_HUMMING_NTAIL="${PREFILL_HUMMING_NTAIL:-0}" \
  -e PREFILL_EP_SIZE="${PREFILL_EP_SIZE:-1}" \
  -e PREFILL_MEM_FRACTION="${PREFILL_MEM_FRACTION:-0.84}" \
  -e SGLANG_EXPERIMENT_DSV41_W2_PAIR_LAYERS="${PREFILL_W2_PAIR_LAYERS:-0}" \
  -e SGLANG_SHARED_EXPERT_TP1="${PREFILL_SHARED_TP1:-0}" \
  -v "${PREFILL_FLASHINFER_SITE:?}:/repro/flashinfer:ro" \
  -v "${PREFILL_EP_PLACEMENT:?}:/repro/ep-placement.json:ro" \
  -v "$model:/models/DeepSeek-V4.1-Flash-1017:ro" \
  -v "$output/source:/repro/source:ro" -v "$output/tools:/repro/tools:ro" \
  -v "$output:/repro/results" -v "$deps:/repro/deps:ro" \
  -v "$cache:/repro/cache" \
  "$image_id" bash /repro/tools/launch_server.sh > "$output/container_id.txt"
owned_container=1
docker inspect "$container" > "$output/container.json"
docker exec "$container" nvidia-smi -q > "$output/nvidia-smi.txt"
python3 "$output/tools/bench.py" --base-url "http://127.0.0.1:$port" \
  --admin-key profile \
  --server-container "$container" \
  --wait-ready "${PREFILL_READY_TIMEOUT:-3600}" --output "$output/benchmark" \
  --profile-dir "$output/profiles" --server-profile-dir /repro/results/profiles \
  --lengths 65536 262144 --warmup 10 --repeat 5 \
  --expected-chunk-size "${PREFILL_CHUNK_SIZE:-16384}" \
  --expected-attention-backend "${PREFILL_ATTENTION_BACKEND:-auto}" \
  --expected-ep-size "${PREFILL_EP_SIZE:-1}" \
  --expected-mem-fraction "${PREFILL_MEM_FRACTION:-0.84}" \
  2>&1 | tee "$output/client.log"
echo "Benchmark complete. Summary: $output/benchmark/summary.md"

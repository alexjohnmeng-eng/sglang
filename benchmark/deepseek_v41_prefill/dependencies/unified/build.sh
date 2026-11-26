#!/usr/bin/env bash
# One source revision + self-contained patch, one wheel, one public module.
set -euo pipefail
output=${1:?Usage: build.sh NEW_OUTPUT_DIR}
assets=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152
[[ ! -e "$output" ]] || { echo "Output exists: $output" >&2; exit 1; }
mkdir -p "$output"
output=$(realpath "$output")
git init "$output/source"
git -C "$output/source" remote add origin https://github.com/deepseek-ai/DeepGEMM.git
git -C "$output/source" fetch --depth 1 origin 724a799bd04382732df8ba73611b4a2702879ac6
git -C "$output/source" checkout --detach FETCH_HEAD
git -C "$output/source" submodule update --init --recursive
git -C "$output/source" apply --unidiff-zero --check "$assets/deepgemm-h20-unified.patch"
git -C "$output/source" apply --unidiff-zero "$assets/deepgemm-h20-unified.patch"
python3 "$assets/verify.py" source "$output/source"
mkdir "$output/wheels" "$output/site"
docker run --rm --network host -v "$output:/build" \
  -e PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  "$image" bash -ec '
    apt-get update -qq
    apt-get install -y -qq libdw-dev
    cd /build/source
    DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 MAX_JOBS=4 /opt/sglang/bin/python3 \
      -m pip wheel --no-deps --no-build-isolation --wheel-dir /build/wheels .
    /opt/sglang/bin/python3 - <<"PY"
from pathlib import Path
from zipfile import ZipFile
wheel, = Path("/build/wheels").glob("*.whl")
with ZipFile(wheel) as archive:
    archive.extractall("/build/site")
PY
  ' 2>&1 | tee "$output/build.log"
python3 "$assets/verify.py" site "$output/site" > "$output/site-hashes.json"

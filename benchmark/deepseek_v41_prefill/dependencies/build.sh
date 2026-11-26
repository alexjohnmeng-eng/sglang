#!/usr/bin/env bash
# Host: build the isolated dense, SM90 MXFP4 MegaMoE and Mega mHC packages.
set -euo pipefail
kind=${1:?Usage: build.sh dense|mega|mhc OUTPUT_DIR}
output=${2:?Choose a new output directory}
assets=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ ! -e "$output" ]] || { echo "Output exists: $output" >&2; exit 1; }
case "$kind" in
  dense) url=https://github.com/sgl-project/DeepGEMM.git; revision=b7f3816c7c0b3468303f198bcfa62eb20d98ee6f; patch=dense-wqb-rope-multicast.patch ;;
  mega) url=https://github.com/deepseek-ai/DeepGEMM.git; revision=724a799bd04382732df8ba73611b4a2702879ac6; patch=sm90-mxfp4-megamoe.patch ;;
  mhc) url=https://github.com/deepseek-ai/DeepGEMM.git; revision=410ce17fb3cc117f0ebdeb8ca0808c5c9264b083; patch=mega-mhc-isolation.patch ;;
  *) echo "Unknown build: $kind" >&2; exit 2 ;;
esac
mkdir -p "$output"
output=$(realpath "$output")
git clone --no-checkout "$url" "$output/source"
git -C "$output/source" checkout --detach "$revision"
git -C "$output/source" submodule update --init --recursive
git -C "$output/source" apply --check "$assets/$patch"
git -C "$output/source" apply "$assets/$patch"
# Require the complete patched-file hashes from the tested source before building.
python3 "$assets/verify.py" source "$kind" "$output/source"
mkdir "$output/site" "$output/wheels"
docker run --rm --network host -v "$output:/build" -e BUILD_KIND="$kind" \
  -e PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
  "${PREFILL_IMAGE:-lmsysorg/sglang:latest}" bash -ec '
    apt-get update -qq
    apt-get install -y -qq libdw-dev
    if [[ "$BUILD_KIND" == dense ]]; then
      mkdir -p /build/site/deep_gemm
      /opt/sglang/bin/python3 /build/source/sgl_deep_gemm/build_ext.py /build/source /build/site/deep_gemm/_C.so
      cp -a /build/source/deep_gemm/. /build/site/deep_gemm/
    else
      cd /build/source
      DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 MAX_JOBS=8 /opt/sglang/bin/python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /build/wheels .
      /opt/sglang/bin/python3 - <<"PY"
from pathlib import Path
import os,zipfile
wheel,=Path("/build/wheels").glob("*.whl")
with zipfile.ZipFile(wheel) as archive: archive.extractall("/build/site")
Path("/build/site/deep_gemm").rename("/build/site/" + ("deep_gemm_mega" if os.environ["BUILD_KIND"] == "mega" else "deep_gemm_mhc"))
PY
    fi
  ' 2>&1 | tee "$output/build.log"
python3 "$assets/verify.py" site "$kind" "$output/site" --report "$output/binary-hashes.json"

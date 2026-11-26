#!/usr/bin/env bash
# Sequential validation: owns one GPU server at a time; never pushes or commits.
set -euo pipefail
site=${1:?Usage: validate.sh UNIFIED_SITE NEW_OUTPUT_ROOT [REVISION_OR_TREE]}
output=${2:?Choose a new output root}
assets=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tools=$(realpath "$assets/../..")
repo=$(git -C "$tools" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse --verify "${3:-HEAD}^{tree}")
site=$(realpath "$site")
python3 "$assets/verify.py" site "$site"
mkdir -- "$output"
output=$(realpath "$output")
printf '%s\n' "$revision" > "$output/source-tree.txt"
export PREFILL_UNIFIED_DEEPGEMM_SITE="$site"
export PREFILL_REVISION="$revision"
for topology in cp8ep8 cp8ep1 dp8ep8; do
  PREFILL_OUTPUT="$output/$topology" bash "$tools/run_h20.sh" "$topology" \
    2>&1 | tee "$output/$topology-driver.log"
done

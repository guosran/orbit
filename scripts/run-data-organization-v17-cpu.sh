#!/usr/bin/env bash
# Compiler-only reproduction; tools must already be built. No NPU calls.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo 'usage: run-data-organization-v17-cpu.sh CORE_BUILD NEW_OUTPUT [AMOEBA_TOOL [AIR_TOOL]]' >&2
  exit 2
fi
orbit_source=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core_build=$(realpath "$1")
result_dir=$(realpath -m "$2")
[[ ! -e "$result_dir" ]] || { echo 'Output must be new.' >&2; exit 2; }
[[ -x "$core_build/tools/taskflow-opt/taskflow-opt" && -x "$core_build/tools/taskflow-opt/taskflow-program-contrast-test" ]] || {
  echo 'Build Core and all test targets first with scripts/build-core.sh.' >&2
  exit 2
}
optional_tools=()
if [[ $# -ge 3 ]]; then optional_tools+=(--amoeba-tool "$(realpath "$3")"); fi
if [[ $# -ge 4 ]]; then optional_tools+=(--air-tool "$(realpath "$4")"); fi
exec python3 "$orbit_source/test/run_core_regressions.py" \
  --tool "$core_build/tools/taskflow-opt/taskflow-opt" \
  --output "$result_dir" "${optional_tools[@]}"

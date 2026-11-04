#!/usr/bin/env bash
# Compatibility entry point: current static planning is C++ only.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$root/scripts/run_taskflow_static_cpp.sh" "$@"

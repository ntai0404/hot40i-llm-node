#!/usr/bin/env bash
set -euo pipefail
echo "NOTE: scripts/build_native.sh is a compatibility alias for build_host.sh" >&2
exec "$(dirname "$0")/build_host.sh" "$@"

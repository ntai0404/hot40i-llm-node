#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-build-android/h40_io_bench}
REMOTE=/data/local/tmp/h40_io_bench
adb push "$BIN" "$REMOTE"
adb shell "chmod 755 $REMOTE"
echo "pushed to $REMOTE"

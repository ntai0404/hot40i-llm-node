#!/usr/bin/env bash
set -euo pipefail
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to the Android NDK root}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-26}"
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

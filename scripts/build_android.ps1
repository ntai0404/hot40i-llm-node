$ErrorActionPreference = "Stop"
if (-not $env:ANDROID_NDK_HOME) { throw "Set ANDROID_NDK_HOME to the Android NDK root" }
$platform = if ($env:ANDROID_PLATFORM) { $env:ANDROID_PLATFORM } else { "android-26" }
$toolchain = Join-Path $env:ANDROID_NDK_HOME "build/cmake/android.toolchain.cmake"
cmake -S . -B build-android `
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM="$platform" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --config Release -j 4

param(
    [string] $SourceDir = "third_party/MNN",
    [string] $BuildDir = "artifacts/build/mnn-android-arm64",
    [string] $NdkHome = "C:/tmp/android-ndk-r27d",
    [string] $AndroidPlatform = "android-26"
)

$ErrorActionPreference = "Stop"

$cmake = "C:/Progra~1/CMake/bin/cmake.exe"
$ninja = (Get-Command ninja.exe -ErrorAction Stop).Source
$toolchain = Join-Path $NdkHome "build/cmake/android.toolchain.cmake"

if (-not (Test-Path -LiteralPath $toolchain)) {
    throw "Android NDK toolchain not found at $toolchain"
}

& $cmake -S $SourceDir -B $BuildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DANDROID_ABI=arm64-v8a" `
    "-DANDROID_PLATFORM=$AndroidPlatform" `
    "-DANDROID_STL=c++_static" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DMNN_BUILD_LLM=ON" `
    "-DMNN_LLM_BUILD_DEMO=ON" `
    "-DLLM_SUPPORT_HTTP_RESOURCE=OFF" `
    "-DMNN_BUILD_FOR_ANDROID_COMMAND=true" `
    "-DMNN_BUILD_SHARED_LIBS=OFF" `
    "-DMNN_SEP_BUILD=OFF" `
    "-DMNN_LOW_MEMORY=ON" `
    "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON" `
    "-DMNN_BUILD_BENCHMARK=OFF" `
    "-DMNN_BUILD_TEST=OFF" `
    "-DMNN_BUILD_OPENCV=OFF" `
    "-DMNN_BUILD_AUDIO=OFF" `
    "-DMNN_BUILD_DIFFUSION=OFF" `
    "-DMNN_OPENCL=OFF" `
    "-DMNN_USE_SSE=OFF"
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed with exit code $LASTEXITCODE"
}

& $cmake --build $BuildDir --target llm_bench -j 4
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed with exit code $LASTEXITCODE"
}

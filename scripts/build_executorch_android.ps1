param(
    [string] $SourceDir = "third_party/executorch",
    [string] $BuildDir = "artifacts/build/executorch-android-arm64",
    [string] $HostFlatbuffersBuildDir = "artifacts/build/flatbuffers-host",
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

$flatbuffersSource = Join-Path $SourceDir "third-party/flatbuffers"
$hostFlatc = Join-Path $HostFlatbuffersBuildDir "flatc.exe"
$hostFlatcAbsolute = (Resolve-Path -LiteralPath $hostFlatc -ErrorAction SilentlyContinue).Path

& $cmake -S $flatbuffersSource -B $HostFlatbuffersBuildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DFLATBUFFERS_BUILD_FLATC=ON" `
    "-DFLATBUFFERS_INSTALL=OFF" `
    "-DFLATBUFFERS_BUILD_FLATHASH=OFF" `
    "-DFLATBUFFERS_BUILD_FLATLIB=OFF" `
    "-DFLATBUFFERS_BUILD_TESTS=OFF"
if ($LASTEXITCODE -ne 0) {
    throw "host flatc configure failed with exit code $LASTEXITCODE"
}

& $cmake --build $HostFlatbuffersBuildDir --target flatc -j 4
if ($LASTEXITCODE -ne 0) {
    throw "host flatc build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $hostFlatc)) {
    throw "host flatc not found at $hostFlatc"
}

$hostFlatcAbsolute = (Resolve-Path -LiteralPath $hostFlatc).Path

& $cmake -S $SourceDir -B $BuildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DANDROID_ABI=arm64-v8a" `
    "-DANDROID_PLATFORM=$AndroidPlatform" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON" `
    "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON" `
    "-DEXECUTORCH_BUILD_XNNPACK=ON" `
    "-DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF" `
    "-DEXECUTORCH_ENABLE_LOGGING=ON" `
    "-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF" `
    "-DFLATC_EXECUTABLE=$hostFlatcAbsolute" `
    "-DPYTHON_EXECUTABLE=python"
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed with exit code $LASTEXITCODE"
}

& $cmake --build $BuildDir --target executor_runner -j 4
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed with exit code $LASTEXITCODE"
}

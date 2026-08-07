param(
    [string] $SourceDir = "third_party/llama.cpp",
    [string] $BuildDir = "artifacts/build/llama-cpp-android-arm64",
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
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DLLAMA_BUILD_COMMIT=e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0" `
    "-DLLAMA_BUILD_NUMBER=10173" `
    "-DBUILD_SHARED_LIBS=OFF" `
    "-DGGML_NATIVE=OFF" `
    "-DGGML_OPENMP=OFF" `
    "-DLLAMA_BUILD_TESTS=OFF" `
    "-DLLAMA_BUILD_EXAMPLES=OFF" `
    "-DLLAMA_BUILD_SERVER=OFF" `
    "-DLLAMA_BUILD_APP=OFF" `
    "-DLLAMA_BUILD_TOOLS=ON" `
    "-DLLAMA_CURL=OFF" `
    "-DLLAMA_OPENSSL=OFF"
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed with exit code $LASTEXITCODE"
}

& $cmake --build $BuildDir --target llama-bench -j 4
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed with exit code $LASTEXITCODE"
}

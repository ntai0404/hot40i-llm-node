param(
    [string] $Source = "scripts/android_gpu_probe.cpp",
    [string] $OutDir = "artifacts/build/mlc-gpu-probe",
    [string] $NdkHome = "C:/tmp/android-ndk-r27d",
    [string] $Api = "26"
)

$ErrorActionPreference = "Stop"

$clang = Join-Path $NdkHome "toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android$Api-clang++.cmd"
if (-not (Test-Path -LiteralPath $clang)) {
    throw "Android clang not found at $clang"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$out = Join-Path $OutDir "gpu_probe"
& $clang $Source -O2 -std=c++17 -ldl -o $out
if ($LASTEXITCODE -ne 0) {
    throw "gpu_probe build failed with exit code $LASTEXITCODE"
}
Get-Item -LiteralPath $out | Select-Object FullName,Length,LastWriteTime

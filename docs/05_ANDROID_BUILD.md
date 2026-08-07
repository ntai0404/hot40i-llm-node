# Android ARM64 build

Use the Android NDK + CMake. Do not assume a generic Linux ARM64 binary is ABI-compatible with Android/Bionic.

## Linux/macOS/WSL

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
# Optional: export ANDROID_PLATFORM=android-26
./scripts/build_android.sh
```

## Windows PowerShell

```powershell
$env:ANDROID_NDK_HOME = "C:\path\to\android-ndk"
# Optional: $env:ANDROID_PLATFORM = "android-26"
.\scripts\build_android.ps1
```

The default API level is only a bootstrap value. D01 must inspect the actual device build and the implementing task may pin a more appropriate minimum after evidence.

For a non-destructive device smoke test, use the repository ADB tooling/push policy or, where the roadmap explicitly calls for it, push test binaries under `/data/local/tmp` and execute them there. Do not write system partitions.

Host C++ smoke builds are primarily validated on Linux CI/WSL. Native Windows is supported for the Python control plane and Android NDK workflow; do not confuse a Windows host compile limitation with an Android ARM64 limitation.

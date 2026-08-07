from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .parsers import (
    count_thermal_zones,
    parse_cpuinfo_hardware,
    parse_cpuinfo_processor_count,
    parse_data_filesystem,
    parse_getprop,
    parse_meminfo,
    parse_storage_roots,
)


class AdbError(RuntimeError):
    pass


class UnsafeDeviceCommand(AdbError):
    pass


@dataclasses.dataclass(slots=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    def check(self) -> "CommandResult":
        if self.returncode != 0:
            raise AdbError(
                f"command failed ({self.returncode}): {' '.join(self.argv)}\n"
                f"stdout:\n{self.stdout}\nstderr:\n{self.stderr}"
            )
        return self


SAFE_PROBES: dict[str, str] = {
    "getprop": "getprop",
    "cpuinfo": "cat /proc/cpuinfo",
    "meminfo": "cat /proc/meminfo",
    "partitions": "cat /proc/partitions",
    "mounts": "cat /proc/mounts",
    "df": "df -h",
    "block_by_name": "ls -l /dev/block/by-name 2>/dev/null || true",
    "thermal_zones": "for z in /sys/class/thermal/thermal_zone*; do echo ==== $z; cat $z/type 2>/dev/null; cat $z/temp 2>/dev/null; done",
    "cpu_online": "cat /sys/devices/system/cpu/online 2>/dev/null || true",
    "cpu_freq": "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ==== $c; cat $c/cpufreq/scaling_cur_freq 2>/dev/null; done",
    "block_devices": "for d in /sys/block/*; do echo ==== $d; cat $d/device/model 2>/dev/null; cat $d/queue/logical_block_size 2>/dev/null; done",
    "storage_sysfs": "for d in /sys/block/sd* /sys/block/mmcblk*; do [ -e $d ] || continue; echo ==== $d; cat $d/device/model 2>/dev/null; cat $d/device/name 2>/dev/null; cat $d/queue/logical_block_size 2>/dev/null; cat $d/queue/physical_block_size 2>/dev/null; cat $d/queue/read_ahead_kb 2>/dev/null; cat $d/queue/rotational 2>/dev/null; cat $d/size 2>/dev/null; done",
    "battery": "dumpsys battery",
}

# Repository wrappers reject destructive patterns. This does not pretend to sandbox
# an agent that bypasses the wrapper and invokes a system executable directly.
BLOCKED_TOKENS = (
    "fastboot", "reboot bootloader", "reboot fastboot", "factory reset", "wipe", "erase ",
    "dd if=", "dd of=/dev", "mkfs", "flash ", "vbmeta", "frp", "/dev/block/",
    "setprop ro.", "recovery --wipe", "rm -rf /data", "rm -rf /system",
)


def _reject_unsafe(text: str) -> None:
    low = " ".join(text.lower().split())
    for token in BLOCKED_TOKENS:
        if token in low:
            raise UnsafeDeviceCommand(f"repository device policy rejected command containing {token!r}")


class AdbClient:
    """Conservative stock-Android ADB wrapper.

    Public operations are intentionally narrow. Arbitrary shell execution is not
    exposed; call sites use named probes or run a previously pushed executable in
    /data/local/tmp. Destructive operations do not exist in this API.
    """

    def __init__(self, serial: str | None = None, adb_path: str = "adb") -> None:
        resolved = shutil.which(adb_path) or (adb_path if Path(adb_path).exists() else None)
        if not resolved:
            raise AdbError("adb was not found in PATH")
        self.adb = resolved
        self.serial = serial

    def _base(self) -> list[str]:
        argv = [self.adb]
        if self.serial:
            argv += ["-s", self.serial]
        return argv

    def _run(self, args: Iterable[str], *, timeout: float = 30, policy_checked: bool = False) -> CommandResult:
        argv = self._base() + list(args)
        if not policy_checked:
            _reject_unsafe(" ".join(argv[1:]))
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return CommandResult(argv, proc.returncode, proc.stdout, proc.stderr)

    def _shell_exact(self, command: str, *, timeout: float = 30) -> CommandResult:
        _reject_unsafe(command)
        return self._run(["shell", command], timeout=timeout, policy_checked=True)

    def probe(self, name: str, *, timeout: float = 45) -> CommandResult:
        try:
            command = SAFE_PROBES[name]
        except KeyError as exc:
            raise AdbError(f"unknown named safe probe: {name}") from exc
        # SAFE_PROBES is an exact, repository-owned allowlist. Some legitimate read-only
        # probes inspect /dev/block metadata, which the generic free-form command filter
        # intentionally rejects. Never route user-provided text through this path.
        return self._run(["shell", command], timeout=timeout, policy_checked=True)

    def devices(self) -> CommandResult:
        return self._run(["devices", "-l"])

    def push(self, local: Path, remote: str, *, timeout: float = 120) -> CommandResult:
        if not (remote == "/data/local/tmp" or remote.startswith("/data/local/tmp/")):
            raise UnsafeDeviceCommand("push destination must be /data/local/tmp during stock-Android phases")
        return self._run(["push", str(local), remote], timeout=timeout)

    def pull(self, remote: str, local: Path, *, timeout: float = 120) -> CommandResult:
        return self._run(["pull", remote, str(local)], timeout=timeout)

    def make_tmp_executable(self, remote_path: str) -> CommandResult:
        if not remote_path.startswith("/data/local/tmp/"):
            raise UnsafeDeviceCommand("only /data/local/tmp files may be chmodded by the repository wrapper")
        command = f"chmod 700 {shlex.quote(remote_path)}"
        return self._shell_exact(command)

    def forward(self, host_port: int, device_port: int) -> CommandResult:
        if not (1 <= host_port <= 65535 and 1 <= device_port <= 65535):
            raise AdbError("invalid TCP port")
        return self._run(["forward", f"tcp:{host_port}", f"tcp:{device_port}"])

    def run_tmp_binary(self, remote_path: str, args: Iterable[str] = (), *, timeout: float = 300) -> CommandResult:
        if not remote_path.startswith("/data/local/tmp/"):
            raise UnsafeDeviceCommand("only /data/local/tmp executables are allowed")
        argv = [remote_path, *list(args)]
        command = " ".join(shlex.quote(x) for x in argv)
        _reject_unsafe(command)
        return self._shell_exact(command, timeout=timeout)

    def run_harmless_workload(self, duration_seconds: int, *, timeout: float | None = None) -> CommandResult:
        if duration_seconds < 1 or duration_seconds > 3600:
            raise AdbError("workload duration must be between 1 and 3600 seconds")
        command = (
            f"end=$((`date +%s`+{duration_seconds})); "
            "while [ `date +%s` -lt $end ]; do "
            "cat /proc/meminfo >/dev/null 2>/dev/null; "
            "cat /system/build.prop >/dev/null 2>/dev/null; "
            "for c in /sys/devices/system/cpu/cpu[0-9]*; do "
            "cat $c/cpufreq/scaling_cur_freq >/dev/null 2>/dev/null; "
            "done; "
            "done"
        )
        return self._run(["shell", command], timeout=timeout or duration_seconds + 30, policy_checked=True)


def collect_manifest(client: AdbClient) -> dict:
    devices = client.devices().check().stdout
    probes: dict[str, dict[str, str | int]] = {}
    for name, command in SAFE_PROBES.items():
        result = client.probe(name)
        probes[name] = {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    props = parse_getprop(str(probes["getprop"]["stdout"]))
    meminfo = parse_meminfo(str(probes["meminfo"]["stdout"]))
    cpuinfo = str(probes["cpuinfo"]["stdout"])
    summary = {
        "product": props.get("ro.product.name"),
        "model": props.get("ro.product.model"),
        "device": props.get("ro.product.device"),
        "manufacturer": props.get("ro.product.manufacturer"),
        "brand": props.get("ro.product.brand"),
        "android_release": props.get("ro.build.version.release"),
        "sdk": props.get("ro.build.version.sdk"),
        "build_fingerprint": props.get("ro.build.fingerprint"),
        "hardware": props.get("ro.boot.hardware") or parse_cpuinfo_hardware(cpuinfo),
        "board_platform": props.get("ro.board.platform"),
        "soc": props.get("ro.boot.auto.efuse") or props.get("ro.boot.auto.chipid"),
        "boot_devices": props.get("ro.boot.boot_devices"),
        "ddrsize": props.get("ro.boot.ddrsize"),
        "slot_suffix": props.get("ro.boot.slot_suffix"),
        "dynamic_partitions": props.get("ro.boot.dynamic_partitions"),
        "verified_boot_state": props.get("ro.boot.verifiedbootstate"),
        "mem_total_bytes": meminfo.get("MemTotal"),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "cpu_online": str(probes["cpu_online"]["stdout"]).strip() or None,
        "processor_count": parse_cpuinfo_processor_count(cpuinfo),
        "storage_roots": parse_storage_roots(str(probes["block_by_name"]["stdout"])),
        "data_filesystem": parse_data_filesystem(str(probes["mounts"]["stdout"])),
        "thermal_zone_count": count_thermal_zones(str(probes["thermal_zones"]["stdout"])),
    }
    return {
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "adb_devices": devices,
        "serial": client.serial,
        "summary": summary,
        "probes": probes,
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

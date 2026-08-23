from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

CANONICAL_ADB = Path(r"C:\mobile-remote-tools\platform-tools\adb.exe")


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceRecord:
    serial: str
    state: str
    properties: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardStatus:
    serial: str
    host_port: int
    device_port: int
    endpoint: str
    endpoint_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TransportUnavailable(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "message": str(self), **self.details}


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
HealthProbe = Callable[[str, float], dict[str, Any]]


def parse_adb_devices(output: str) -> list[DeviceRecord]:
    records: list[DeviceRecord] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("List of devices attached") or stripped.startswith("*"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            continue
        properties: dict[str, str] = {}
        for field in fields[2:]:
            if ":" in field:
                key, value = field.split(":", 1)
                properties[key] = value
        records.append(DeviceRecord(serial=fields[0], state=fields[1], properties=properties))
    return records


def _default_runner(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _default_health_probe(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise TransportUnavailable(
                    "endpoint_unhealthy", f"device health returned HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except TransportUnavailable:
        raise
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise TransportUnavailable("endpoint_unhealthy", f"device health failed: {exc}") from exc
    if payload.get("status") != "ok":
        raise TransportUnavailable(
            "endpoint_unhealthy", "device health payload did not report status=ok", payload=payload
        )
    return payload


class AdbForwardSupervisor:
    """Pin one authorized USB device and maintain a verified localhost forward."""

    def __init__(
        self,
        *,
        expected_serial: str | None = None,
        host_port: int = 18080,
        device_port: int = 8080,
        adb_path: str | Path | None = None,
        command_runner: CommandRunner = _default_runner,
        health_probe: HealthProbe = _default_health_probe,
    ) -> None:
        if not (1 <= host_port <= 65535 and 1 <= device_port <= 65535):
            raise ValueError("invalid TCP port")
        selected_adb = adb_path or (CANONICAL_ADB if CANONICAL_ADB.exists() else "adb")
        self.adb = str(selected_adb)
        self.expected_serial = expected_serial
        self.host_port = host_port
        self.device_port = device_port
        self.endpoint = f"http://127.0.0.1:{host_port}/health"
        self._command_runner = command_runner
        self._health_probe = health_probe

    def _run(self, *args: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
        result = self._command_runner([self.adb, *args], timeout)
        if result.returncode != 0:
            raise TransportUnavailable(
                "adb_error",
                f"adb command failed ({result.returncode}): {' '.join(args)}",
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    def devices(self) -> list[DeviceRecord]:
        return parse_adb_devices(self._run("devices", "-l").stdout)

    def _select_device(self) -> DeviceRecord:
        records = self.devices()
        ready = [record for record in records if record.state == "device"]
        if self.expected_serial:
            exact = next((record for record in records if record.serial == self.expected_serial), None)
            if exact and exact.state == "device":
                return exact
            if exact:
                raise TransportUnavailable(
                    "device_not_ready",
                    f"expected device {self.expected_serial} is {exact.state}",
                    serial=self.expected_serial,
                    state=exact.state,
                )
            if ready:
                raise TransportUnavailable(
                    "serial_changed",
                    f"expected {self.expected_serial}, found {ready[0].serial}",
                    expected_serial=self.expected_serial,
                    discovered_serials=[record.serial for record in ready],
                )
            raise TransportUnavailable(
                "no_device", f"expected device {self.expected_serial} is absent", records=[
                    dataclasses.asdict(record) for record in records
                ]
            )
        if not ready:
            code = "device_not_ready" if records else "no_device"
            raise TransportUnavailable(
                code,
                "no authorized ADB device is ready",
                records=[dataclasses.asdict(record) for record in records],
            )
        if len(ready) > 1:
            raise TransportUnavailable(
                "multiple_devices",
                "multiple authorized ADB devices require an explicit serial",
                discovered_serials=[record.serial for record in ready],
            )
        self.expected_serial = ready[0].serial
        return ready[0]

    def _forward_lines(self) -> list[str]:
        return [line.strip() for line in self._run("forward", "--list").stdout.splitlines() if line.strip()]

    def inspect(self) -> ForwardStatus:
        device = self._select_device()
        expected = f"{device.serial} tcp:{self.host_port} tcp:{self.device_port}"
        if expected not in self._forward_lines():
            raise TransportUnavailable(
                "forward_missing",
                f"missing {expected}",
                serial=device.serial,
                forwards=self._forward_lines(),
            )
        payload = self._health_probe(self.endpoint, 3.0)
        return ForwardStatus(
            serial=device.serial,
            host_port=self.host_port,
            device_port=self.device_port,
            endpoint=self.endpoint,
            endpoint_payload=payload,
        )

    def ensure(self) -> ForwardStatus:
        device = self._select_device()
        self._run(
            "-s",
            device.serial,
            "forward",
            f"tcp:{self.host_port}",
            f"tcp:{self.device_port}",
        )
        return self.inspect()

    def remove_forward(self) -> None:
        device = self._select_device()
        self._run("-s", device.serial, "forward", "--remove", f"tcp:{self.host_port}")

    def restart_adb_server(self, *, terminate_stale: bool = False) -> None:
        if terminate_stale and os.name == "nt" and self._command_runner is _default_runner:
            subprocess.run(
                ["taskkill", "/F", "/IM", "adb.exe"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        else:
            try:
                self._run("kill-server")
            except TransportUnavailable:
                pass
        self._run("start-server", timeout=30.0)

    def recover(self, *, attempts: int = 4) -> ForwardStatus:
        failures: list[dict[str, Any]] = []
        try:
            return self.ensure()
        except TransportUnavailable as exc:
            if exc.code == "serial_changed":
                raise
            failures.append(exc.to_dict())
            if exc.code in {"no_device", "device_not_ready", "adb_error"}:
                self.restart_adb_server(terminate_stale=exc.code == "no_device")
        for attempt in range(attempts):
            time.sleep(0.25 * (attempt + 1))
            try:
                return self.ensure()
            except TransportUnavailable as exc:
                if exc.code == "serial_changed":
                    raise
                failures.append(exc.to_dict())
        raise TransportUnavailable(
            "recovery_failed",
            f"ADB forward recovery failed after {attempts + 1} attempts",
            failures=failures,
        )


def resolve_adb_path() -> str:
    if CANONICAL_ADB.exists():
        return str(CANONICAL_ADB)
    return shutil.which("adb") or "adb"

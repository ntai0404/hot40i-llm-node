from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from host.gateway.app import create_app
from host.gateway.device_client import HttpDeviceTokenClient
from host.transport.adb_forward import (
    AdbForwardSupervisor,
    TransportUnavailable,
    parse_adb_devices,
)


class FakeAdb:
    def __init__(self, serial: str = "SERIAL1", state: str = "device") -> None:
        self.serial = serial
        self.state = state
        self.forwarded = False
        self.commands: list[list[str]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        command = list(argv)
        self.commands.append(command)
        args = command[1:]
        if args == ["devices", "-l"]:
            output = "List of devices attached\n"
            if self.serial:
                output += f"{self.serial} {self.state} product:X model:Y transport_id:2\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if args == ["forward", "--list"]:
            output = f"{self.serial} tcp:18080 tcp:8080\n" if self.forwarded else ""
            return subprocess.CompletedProcess(command, 0, output, "")
        if args[-3:] == ["forward", "tcp:18080", "tcp:8080"]:
            self.forwarded = True
            return subprocess.CompletedProcess(command, 0, "18080\n", "")
        if args[-3:] == ["forward", "--remove", "tcp:18080"]:
            self.forwarded = False
            return subprocess.CompletedProcess(command, 0, "", "")
        if args in (["kill-server"], ["start-server"]):
            if args == ["kill-server"]:
                self.forwarded = False
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", f"unexpected command: {args}")


def make_supervisor(fake: FakeAdb, expected_serial: str | None = None) -> AdbForwardSupervisor:
    return AdbForwardSupervisor(
        expected_serial=expected_serial,
        adb_path="adb-test",
        command_runner=fake,
        health_probe=lambda _url, _timeout: {"status": "ok"},
    )


def test_parse_adb_devices_states_and_properties() -> None:
    records = parse_adb_devices(
        "List of devices attached\n"
        "SERIAL1 device product:X6528-OP model:Infinix_X6528 transport_id:2\n"
        "SERIAL2 unauthorized usb:1-2 transport_id:3\n"
    )
    assert [(record.serial, record.state) for record in records] == [
        ("SERIAL1", "device"),
        ("SERIAL2", "unauthorized"),
    ]
    assert records[0].properties["model"] == "Infinix_X6528"


def test_ensure_pins_serial_and_recovers_removed_forward() -> None:
    fake = FakeAdb()
    supervisor = make_supervisor(fake)
    assert supervisor.ensure().serial == "SERIAL1"
    assert supervisor.expected_serial == "SERIAL1"
    supervisor.remove_forward()
    with pytest.raises(TransportUnavailable, match="missing") as missing:
        supervisor.inspect()
    assert missing.value.code == "forward_missing"
    assert supervisor.recover().endpoint_payload == {"status": "ok"}


def test_serial_change_and_unauthorized_are_explicit() -> None:
    changed = make_supervisor(FakeAdb(serial="SERIAL2"), expected_serial="SERIAL1")
    with pytest.raises(TransportUnavailable) as serial_error:
        changed.ensure()
    assert serial_error.value.code == "serial_changed"
    assert serial_error.value.details["discovered_serials"] == ["SERIAL2"]

    unauthorized = make_supervisor(
        FakeAdb(serial="SERIAL1", state="unauthorized"), expected_serial="SERIAL1"
    )
    with pytest.raises(TransportUnavailable) as state_error:
        unauthorized.ensure()
    assert state_error.value.code == "device_not_ready"
    assert state_error.value.details["state"] == "unauthorized"


def test_device_client_recovers_once_after_transport_drop() -> None:
    class Guard:
        ensures = 0
        recoveries = 0

        def ensure(self) -> None:
            self.ensures += 1

        def recover(self) -> None:
            self.recoveries += 1

    guard = Guard()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("simulated forward drop", request=request)
        return httpx.Response(200, json={"status": "ok"})

    async def run() -> dict:
        client = HttpDeviceTokenClient(
            "http://device",
            transport=httpx.MockTransport(handler),
            transport_guard=guard,
        )
        return await client.health()

    assert asyncio.run(run()) == {"status": "ok"}
    assert (guard.ensures, guard.recoveries, calls) == (1, 1, 2)


def test_transport_health_returns_explicit_failure() -> None:
    class Device:
        async def health(self) -> dict:
            return {"status": "ok"}

    class ChangedSupervisor:
        def inspect(self) -> None:
            raise TransportUnavailable(
                "serial_changed", "expected SERIAL1, found SERIAL2", expected_serial="SERIAL1"
            )

    app = create_app(device_client=Device())
    app.state.transport_supervisor = ChangedSupervisor()
    response = TestClient(app).get("/transport/health")
    assert response.status_code == 503
    assert response.json()["code"] == "serial_changed"

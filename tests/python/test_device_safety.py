from pathlib import Path
import pytest
from host.device_lab.adb import AdbClient, UnsafeDeviceCommand, _reject_unsafe


def test_destructive_tokens_rejected():
    for cmd in ["fastboot flash boot x.img", "reboot bootloader", "dd of=/dev/block/foo", "rm -rf /data", "fastboot erase userdata"]:
        with pytest.raises(UnsafeDeviceCommand):
            _reject_unsafe(cmd)


def test_safe_probe_text_allowed():
    _reject_unsafe("cat /proc/meminfo")


def test_push_guard_without_invoking_adb(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: "/bin/true")
    c=AdbClient()
    f=tmp_path/'x'; f.write_text('x')
    with pytest.raises(UnsafeDeviceCommand):
        c.push(f, "/sdcard/x")


def test_allowlisted_block_metadata_probe_is_not_false_positive(monkeypatch):
    """The exact read-only allowlist may inspect block metadata without enabling free-form access."""
    monkeypatch.setattr("shutil.which", lambda _: "/bin/true")
    client = AdbClient(adb_path="adb")
    result = client.probe("block_by_name")
    assert result.returncode == 0


def test_free_form_block_device_text_remains_rejected():
    with pytest.raises(UnsafeDeviceCommand):
        _reject_unsafe("ls -l /dev/block/by-name")


def test_make_tmp_executable_guard(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/bin/true")
    client = AdbClient()
    assert client.make_tmp_executable("/data/local/tmp/h40_io_bench").returncode == 0
    with pytest.raises(UnsafeDeviceCommand):
        client.make_tmp_executable("/sdcard/h40_io_bench")

from __future__ import annotations

import re


def parse_meminfo(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB$", line.strip())
        if match:
            result[match.group(1)] = int(match.group(2)) * 1024
    return result


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def parse_getprop(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\[(?P<key>[^\]]+)\]: \[(?P<value>.*)\]$", line.strip())
        if match:
            result[match.group("key")] = match.group("value")
    return result


def parse_cpuinfo_processor_count(text: str) -> int | None:
    processors = {
        match.group(1)
        for match in re.finditer(r"^processor\s*:\s*(\d+)\s*$", text, re.MULTILINE)
    }
    return len(processors) if processors else None


def parse_cpuinfo_hardware(text: str) -> str | None:
    match = re.search(r"^Hardware\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_storage_roots(block_by_name: str) -> list[str]:
    roots: set[str] = set()
    for match in re.finditer(r"-> /dev/block/([a-zA-Z]+)", block_by_name):
        roots.add(match.group(1))
    return sorted(roots)


def parse_data_filesystem(mounts: str) -> str | None:
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "/data":
            return parts[2]
    return None


def count_thermal_zones(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("==== /sys/class/thermal/thermal_zone"))

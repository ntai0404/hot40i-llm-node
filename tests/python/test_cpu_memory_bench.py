from __future__ import annotations

from host.device_lab.cpu_memory import parse_cpu_topology, select_configs


def test_selects_big_little_configs_from_topology_probe() -> None:
    raw = "\n".join(
        [
            "==== /sys/devices/system/cpu/cpu0",
            "0",
            "0",
            "1600000",
            "800000",
            "128",
            "==== /sys/devices/system/cpu/cpu1",
            "1",
            "0",
            "1600000",
            "800000",
            "128",
            "==== /sys/devices/system/cpu/cpu6",
            "6",
            "0",
            "2000000",
            "1200000",
            "512",
            "==== /sys/devices/system/cpu/cpu7",
            "7",
            "0",
            "2000000",
            "1200000",
            "512",
        ]
    )
    topology = parse_cpu_topology(raw)
    configs = {item["name"]: item for item in select_configs(topology)}

    assert configs["a75_focused"]["cpus"] == [6, 7]
    assert configs["a75_focused"]["threads"] == 2
    assert configs["a55_focused"]["cpus"] == [0, 1]
    assert configs["mixed_all_core"]["cpus"] == [0, 1, 6, 7]

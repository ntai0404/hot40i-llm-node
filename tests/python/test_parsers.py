from host.device_lab.parsers import parse_meminfo


def test_parse_meminfo() -> None:
    data = parse_meminfo("MemTotal:        4000000 kB\nMemAvailable:    2000000 kB\n")
    assert data["MemTotal"] == 4_000_000 * 1024
    assert data["MemAvailable"] == 2_000_000 * 1024

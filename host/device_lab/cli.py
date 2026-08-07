from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adb import AdbClient, AdbError, collect_manifest, save_json
from .parsers import human_bytes, parse_meminfo

app = typer.Typer(no_args_is_help=True, help="Safe Hot 40i device laboratory")
console = Console()


@app.command()
def doctor() -> None:
    """Check laptop-side prerequisites without changing the device."""
    rows = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "adb": shutil.which("adb") or "MISSING",
        "cmake": shutil.which("cmake") or "MISSING",
        "git": shutil.which("git") or "MISSING",
    }
    table = Table(title="Hot40i lab doctor")
    table.add_column("check")
    table.add_column("value")
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)


@app.command()
def devices() -> None:
    client = AdbClient()
    console.print(client.devices().check().stdout)


@app.command()
def probe(
    out: Path = typer.Option(Path("artifacts/device-manifest.json"), "--out"),
    raw_dir: Path | None = typer.Option(None, "--raw-dir"),
    serial: str | None = typer.Option(None, "--serial"),
) -> None:
    """Collect a read-only hardware/software inventory from Android."""
    try:
        client = AdbClient(serial=serial)
        manifest = collect_manifest(client)
        save_json(manifest, out)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            for name, result in manifest["probes"].items():
                (raw_dir / f"{name}.stdout.txt").write_text(str(result["stdout"]), encoding="utf-8")
                (raw_dir / f"{name}.stderr.txt").write_text(str(result["stderr"]), encoding="utf-8")
    except AdbError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]wrote[/green] {out}")


@app.command()
def mem(serial: str | None = typer.Option(None, "--serial")) -> None:
    client = AdbClient(serial=serial)
    raw = client.probe("meminfo").check().stdout
    info = parse_meminfo(raw)
    for key in ["MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"]:
        if key in info:
            console.print(f"{key:14} {human_bytes(info[key])}")


@app.command("forward")
def forward_port(
    host_port: int = typer.Option(18080),
    device_port: int = typer.Option(8080),
    serial: str | None = typer.Option(None, "--serial"),
) -> None:
    client = AdbClient(serial=serial)
    client.forward(host_port, device_port).check()
    console.print(f"localhost:{host_port} -> device:{device_port} over ADB")


@app.command("show-manifest")
def show_manifest(path: Path = Path("artifacts/device-manifest.json")) -> None:
    console.print_json(json=path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()

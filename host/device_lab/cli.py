from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adb import AdbClient, AdbError, collect_manifest, save_json
from .cpu_memory import run_cpu_memory_benchmark
from .io_bench import run_storage_benchmark
from .memory_budget import measure_memory_budget
from .parsers import human_bytes, parse_meminfo
from .storage_identity import identify_storage
from .thermal import collect_thermal_baseline

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


@app.command("memory-budget")
def memory_budget(
    out: Path = typer.Option(Path("benchmarks/stock/memory_budget.json"), "--out"),
    samples: int = typer.Option(3, "--samples"),
    interval: float = typer.Option(5.0, "--interval"),
    serial: str | None = typer.Option(None, "--serial"),
    manifest: Path = typer.Option(Path("artifacts/device-manifest.json"), "--manifest"),
) -> None:
    budget = measure_memory_budget(
        out=out,
        samples_count=samples,
        interval_seconds=interval,
        serial=serial,
        manifest_path=manifest,
    )
    console.print(f"safe_rss_budget_bytes={budget}")


@app.command("storage-identity")
def storage_identity(
    out: Path = typer.Option(Path("benchmarks/stock/storage_identity.json"), "--out"),
    serial: str | None = typer.Option(None, "--serial"),
    manifest: Path = typer.Option(Path("artifacts/device-manifest.json"), "--manifest"),
) -> None:
    storage_class = identify_storage(out=out, serial=serial, manifest_path=manifest)
    console.print(f"storage_classification={storage_class}")


@app.command("thermal-baseline")
def thermal_baseline(
    serial: str = typer.Option(..., "--serial"),
    duration: int = typer.Option(900, "--duration"),
    interval: float = typer.Option(5.0, "--interval"),
    thermal_jsonl: Path = typer.Option(Path("artifacts/runs/thermal.jsonl"), "--thermal-jsonl"),
    out: Path = typer.Option(Path("benchmarks/stock/thermal_baseline.json"), "--out"),
) -> None:
    collect_thermal_baseline(
        serial=serial,
        duration_seconds=duration,
        interval_seconds=interval,
        thermal_jsonl=thermal_jsonl,
        summary_out=out,
    )
    console.print(f"wrote {out} and {thermal_jsonl}")


@app.command("storage-bench")
def storage_bench(
    serial: str = typer.Option(..., "--serial"),
    binary: Path = typer.Option(Path("build-android/h40_io_bench"), "--binary"),
    samples_jsonl: Path = typer.Option(Path("artifacts/runs/storage_samples.jsonl"), "--samples-jsonl"),
    out: Path = typer.Option(Path("benchmarks/stock/storage.json"), "--out"),
    manifest: Path = typer.Option(Path("artifacts/device-manifest.json"), "--manifest"),
    repeats: int = typer.Option(3, "--repeats"),
    file_size_mib: int = typer.Option(512, "--file-size-mib"),
) -> None:
    document = run_storage_benchmark(
        serial=serial,
        local_binary=binary,
        manifest_path=manifest,
        samples_jsonl=samples_jsonl,
        out=out,
        repeats=repeats,
        file_size_mib=file_size_mib,
    )
    console.print(
        f"wrote {out} with {document['metrics']['sample_count']} samples; "
        f"samples={samples_jsonl}"
    )


@app.command("cpu-memory-bench")
def cpu_memory_bench(
    serial: str = typer.Option(..., "--serial"),
    binary: Path = typer.Option(Path("build-android-direct/h40_cpu_memory_bench"), "--binary"),
    samples_jsonl: Path = typer.Option(Path("artifacts/runs/cpu_memory_samples.jsonl"), "--samples-jsonl"),
    out: Path = typer.Option(Path("benchmarks/stock/cpu_memory.json"), "--out"),
    repeats: int = typer.Option(3, "--repeats"),
    seconds: float = typer.Option(0.35, "--seconds"),
    mem_bytes: int = typer.Option(32 * 1024 * 1024, "--mem-bytes"),
) -> None:
    document = run_cpu_memory_benchmark(
        serial=serial,
        local_binary=binary,
        out=out,
        samples_jsonl=samples_jsonl,
        repeats=repeats,
        seconds=seconds,
        mem_bytes=mem_bytes,
    )
    console.print(
        f"wrote {out} with {document['metrics']['sample_count']} samples; "
        f"samples={samples_jsonl}"
    )


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

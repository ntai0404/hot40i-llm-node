$ErrorActionPreference = "Stop"
$python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe scripts\handoff_check.py --quick
& .\.venv\Scripts\python.exe scripts\taskctl.py status

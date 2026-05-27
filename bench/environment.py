import os
import platform
import subprocess
from pathlib import Path

import duckdb

from config import PROJECT_ROOT


def _run_stdout(args: list[str], cwd: Path | None = None) -> str | None:
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_commit() -> str:
    return _run_stdout(["git", "rev-parse", "HEAD"], PROJECT_ROOT) or "unknown"


def get_git_dirty() -> bool:
    status = _run_stdout(["git", "status", "--porcelain"], PROJECT_ROOT)
    return bool(status)


def get_cpu_model() -> str:
    if platform.system() == "Darwin":
        return _run_stdout(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"

    if platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            with cpuinfo.open() as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()

    return "unknown"


def collect_environment(seed: int, sizes: list[str], targets: list[dict]) -> dict:
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_model": get_cpu_model(),
        "cpu_cores": os.cpu_count(),
        "duckdb_python_package_version": duckdb.__version__,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "targets": targets,
        "dataset": {
            "seed": seed,
            "sizes": sizes,
            "schema": "marketing_telemetry_v2_archetypes",
        },
    }

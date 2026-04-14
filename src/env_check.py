"""Environment inspection helpers for local LLM evaluation."""

from __future__ import annotations

import importlib.util
import logging
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil
import requests

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeInfo:
    """Python and operating system information."""

    os_name: str
    os_version: str
    python_version: str
    python_executable: str


@dataclass(slots=True)
class HardwareInfo:
    """Basic local hardware information."""

    cpu_name: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    ram_gb: float


@dataclass(slots=True)
class GpuInfo:
    """GPU details from torch or nvidia-smi."""

    source: str
    gpu_count: int
    gpus: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class BackendStatus:
    """Status for an execution backend."""

    installed: bool
    reachable: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentSummary:
    """Full environment summary for display and result metadata."""

    runtime: RuntimeInfo
    hardware: HardwareInfo
    torch_cuda: dict[str, Any]
    nvidia_smi: dict[str, Any]
    backends: dict[str, BackendStatus]


def _run_command(args: list[str]) -> tuple[bool, str]:
    """Run a subprocess and capture stdout."""
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, str(exc)
    return True, result.stdout.strip()


def detect_torch_cuda() -> dict[str, Any]:
    """Inspect CUDA availability via torch if torch is installed."""
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "cuda_available": False, "gpu_count": 0, "gpus": []}

    import torch  # type: ignore

    gpus: list[dict[str, Any]] = []
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0

    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "index": index,
            }
        )

    return {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpus": gpus,
    }


def detect_nvidia_smi() -> dict[str, Any]:
    """Inspect GPU information from nvidia-smi if available."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"installed": False, "gpu_count": 0, "gpus": []}

    ok, output = _run_command(
        [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not ok:
        return {"installed": True, "gpu_count": 0, "gpus": [], "error": output}

    gpus: list[dict[str, Any]] = []
    for index, line in enumerate(output.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            memory_gb = round(float(parts[1]) / 1024.0, 2)
            gpus.append({"index": index, "name": parts[0], "total_memory_gb": memory_gb})
    return {"installed": True, "gpu_count": len(gpus), "gpus": gpus}


def detect_ollama(base_url: str) -> BackendStatus:
    """Detect whether Ollama is installed locally and whether its API is reachable."""
    installed = shutil.which("ollama") is not None
    reachable = False
    details: dict[str, Any] = {"base_url": base_url}
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        reachable = response.ok
        if reachable:
            details["models_detected"] = len((response.json() or {}).get("models", []))
    except requests.RequestException as exc:
        details["error"] = str(exc)
    return BackendStatus(installed=installed, reachable=reachable, details=details)


def detect_vllm(base_url: str) -> BackendStatus:
    """Detect whether vLLM is installed and whether a local server seems reachable."""
    installed = importlib.util.find_spec("vllm") is not None
    reachable = False
    details: dict[str, Any] = {"base_url": base_url}
    for path in ("/health", "/v1/models"):
        try:
            response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=3)
            if response.ok:
                reachable = True
                break
        except requests.RequestException as exc:
            details["error"] = str(exc)
    return BackendStatus(installed=installed, reachable=reachable, details=details)


def build_environment_summary(
    ollama_url: str = "http://localhost:11434",
    vllm_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """Build a complete summary dictionary for the current machine."""
    runtime = RuntimeInfo(
        os_name=platform.system(),
        os_version=platform.platform(),
        python_version=platform.python_version(),
        python_executable=shutil.which("python") or "unknown",
    )
    hardware = HardwareInfo(
        cpu_name=platform.processor() or platform.machine() or "unknown",
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False),
        ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
    )
    summary = EnvironmentSummary(
        runtime=runtime,
        hardware=hardware,
        torch_cuda=detect_torch_cuda(),
        nvidia_smi=detect_nvidia_smi(),
        backends={
            "ollama": detect_ollama(ollama_url),
            "vllm": detect_vllm(vllm_url),
        },
    )
    return asdict(summary)


def print_environment_summary(summary: dict[str, Any]) -> None:
    """Print the environment summary in a compact readable format."""
    runtime = summary["runtime"]
    hardware = summary["hardware"]
    torch_cuda = summary["torch_cuda"]
    nvidia_smi = summary["nvidia_smi"]
    backends = summary["backends"]

    print("Environment Summary")
    print(f"  OS: {runtime['os_name']} ({runtime['os_version']})")
    print(f"  Python: {runtime['python_version']} [{runtime['python_executable']}]")
    print(
        f"  CPU: {hardware['cpu_name']} | logical={hardware['cpu_count_logical']} "
        f"physical={hardware['cpu_count_physical']}"
    )
    print(f"  RAM: {hardware['ram_gb']} GB")
    print(
        f"  torch CUDA: installed={torch_cuda.get('installed')} "
        f"available={torch_cuda.get('cuda_available')} count={torch_cuda.get('gpu_count')}"
    )
    if torch_cuda.get("gpus"):
        for gpu in torch_cuda["gpus"]:
            print(f"    torch GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    print(
        f"  nvidia-smi: installed={nvidia_smi.get('installed')} "
        f"count={nvidia_smi.get('gpu_count')}"
    )
    for gpu in nvidia_smi.get("gpus", []):
        print(f"    nvidia-smi GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    for backend_name, status in backends.items():
        print(
            f"  {backend_name}: installed={status['installed']} "
            f"reachable={status['reachable']} details={status['details']}"
        )


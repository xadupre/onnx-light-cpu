"""Private file protocol and single-runtime workers for benchmark_avx2_parity."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROTOCOL_VERSION = 1
RUNTIMES = ("onnx-light-cpu", "onnxruntime")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_snapshot() -> dict[str, dict[str, Any]]:
    """Pin actual imported modules, bypassing stale editable-install redirects."""
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("onnx_light.", "onnx_light_cpu.")) and name not in {
            "onnx_light",
            "onnx_light_cpu",
        }:
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        native = any(str(path).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)
        modules[name] = {
            "path": str(path),
            "package_paths": (list(module.__path__) if hasattr(module, "__path__") else None),
            "sha256": file_digest(path) if native else None,
        }
    if not any(
        name.startswith("onnx_light_cpu.") and entry["sha256"] for name, entry in modules.items()
    ):
        raise RuntimeError("No loaded CPU native modules to pin for workers")
    return modules


def shared_library_snapshot() -> dict[str, str] | None:
    """Include runtime DSOs: unchanged Python extensions may link different kernels."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return None
    paths = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        filename = fields[5]
        if not Path(filename).name.startswith("liblib_onnx"):
            continue
        if filename.endswith(" (deleted)"):
            raise RuntimeError(f"Loaded runtime library was replaced: {filename}")
        paths.add(Path(filename).resolve())
    return {str(path): file_digest(path) for path in sorted(paths)}


def verify_shared_libraries(expected: dict[str, str] | None) -> None:
    if expected is None:
        return
    actual = shared_library_snapshot()
    if actual is None or any(actual.get(path) != digest for path, digest in expected.items()):
        raise RuntimeError("Loaded runtime shared libraries differ from the parent snapshot")


class _PinnedImports:
    def __init__(self, modules: dict[str, dict[str, Any]]):
        self.modules = modules

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        entry = self.modules.get(fullname)
        if entry is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname, entry["path"], submodule_search_locations=entry["package_paths"]
        )


def verify_native_modules(modules: dict[str, dict[str, Any]]) -> None:
    for name, entry in modules.items():
        if entry["sha256"] is None:
            continue
        if file_digest(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"Native module changed on disk: {name}")
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        if str(actual) != entry["path"] or file_digest(actual) != entry["sha256"]:
            raise RuntimeError(f"Native module mismatch for {name}: {actual}")


def write_fixture(case: Any, directory: Path) -> dict[str, Any]:
    """Serialize once; both runtimes read exactly the same model and tensor bytes."""
    model = case.model
    model_path = directory / "model.onnx"
    model_path.write_bytes(model.SerializeToString())
    files = {"model.onnx": file_digest(model_path)}
    datasets = []
    names = [value.name for value in model.graph.input]
    for index, dataset in enumerate(case.data_sets):
        tensors = []
        for number, (name, tensor) in enumerate(zip(names, dataset.inputs, strict=True)):
            filename = f"input-{index}-{number}.bin"
            path = directory / filename
            path.write_bytes(tensor.raw_data())
            files[filename] = file_digest(path)
            tensors.append(
                {
                    "name": name,
                    "file": filename,
                    "data_type": int(tensor.data_type),
                    "shape": list(tensor.shape),
                }
            )
        datasets.append(tensors)
    if not datasets:
        raise ValueError(f"{case.name}: no backend input datasets")
    fixture = {"case": case.name, "datasets": datasets, "files": files}
    fixture["sha256"] = hashlib.sha256(
        json.dumps(fixture, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fixture


def _load_inputs(request: dict[str, Any]) -> list[Any]:
    directory = Path(request["directory"])
    fixture = request["fixture"]
    unsigned = {key: value for key, value in fixture.items() if key != "sha256"}
    if (
        hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        != fixture["sha256"]
    ):
        raise RuntimeError("Fixture manifest checksum mismatch")
    for filename, digest in fixture["files"].items():
        if file_digest(directory / filename) != digest:
            raise RuntimeError(f"Fixture checksum mismatch: {filename}")
    datasets = []
    for dataset in fixture["datasets"]:
        inputs = []
        for tensor in dataset:
            data = (directory / tensor["file"]).read_bytes()
            inputs.append(
                SimpleNamespace(
                    data_type=tensor["data_type"],
                    shape=tensor["shape"],
                    raw_data=lambda data=data: data,
                )
            )
        datasets.append(SimpleNamespace(inputs=inputs))
    return datasets


def _measure_ort(model: bytes, feeds: list[Any], request: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime  # pyrefly: ignore[missing-import]
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        Fail,
        InvalidArgument,
        InvalidGraph,
        NotImplemented,
        RuntimeException,
    )

    try:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = request["threads"]
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session = onnxruntime.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )
        for feed in feeds:
            session.run(None, feed)
    except (Fail, InvalidArgument, InvalidGraph, NotImplemented, RuntimeException) as exc:
        return {"durations": [], "error": str(exc) or type(exc).__name__}

    warmup_start = time.perf_counter()
    for _ in range(request["warmup"]):
        for feed in feeds:
            session.run(None, feed)
        if time.perf_counter() - warmup_start >= request["max_repeat_time"]:
            break
    durations = []
    repeat_start = time.perf_counter()
    for _ in range(request["repeat"]):
        start = time.perf_counter_ns()
        for feed in feeds:
            session.run(None, feed)
        durations.append((time.perf_counter_ns() - start) / 1_000_000_000)
        if time.perf_counter() - repeat_start >= request["max_repeat_time"]:
            break
    return {"durations": durations, "error": None}


def execute_worker(request: dict[str, Any]) -> dict[str, Any]:
    if request["version"] != PROTOCOL_VERSION or request["runtime"] not in RUNTIMES:
        raise ValueError("Unknown AVX2 worker protocol or runtime")
    datasets = _load_inputs(request)
    model_bytes = (Path(request["directory"]) / "model.onnx").read_bytes()
    if request["runtime"] == "onnx-light-cpu":
        from onnx_light.onnx import ModelProto  # pyrefly: ignore[missing-import]
        from onnx_light_cpu import detect_simd_level, register_kernels
        from onnx_light_cpu._benchmark import _measure_case

        register_kernels()
        verify_native_modules(request["modules"])
        verify_shared_libraries(request.get("shared_libraries"))
        if detect_simd_level().name != "AVX2":
            raise RuntimeError("CPU worker does not detect AVX2")
        model = ModelProto()
        model.ParseFromString(model_bytes)
        case = SimpleNamespace(name=request["fixture"]["case"], model=model, data_sets=datasets)
        raw, aggregate = _measure_case(
            case,
            repeat=request["repeat"],
            warmup=request["warmup"],
            max_repeat_time=request["max_repeat_time"],
            threads=request["threads"],
            with_onnxruntime=False,
        )
        verify_native_modules(request["modules"])
        verify_shared_libraries(request.get("shared_libraries"))
        payload = {"raw": raw, "aggregate": aggregate}
    else:
        # Only dtype conversion is shared; this does not create a CPU evaluator or pool.
        from onnx_light_cpu._benchmark import _to_numpy

        feeds = [
            {
                tensor["name"]: _to_numpy(value)
                for tensor, value in zip(description, dataset.inputs, strict=True)
            }
            for description, dataset in zip(request["fixture"]["datasets"], datasets, strict=True)
        ]
        payload = _measure_ort(model_bytes, feeds, request)
    return {
        "version": PROTOCOL_VERSION,
        "runtime": request["runtime"],
        "case": request["fixture"]["case"],
        "fixture_sha256": request["fixture"]["sha256"],
        "threads": request["threads"],
        "payload": payload,
    }


def run_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Wait for process exit, including destruction of its runtime's thread pools."""
    directory = Path(request["directory"])
    request_path = directory / "request.json"
    result_path = directory / "result.json"
    result_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(request_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    context = f"{request['fixture']['case']} ({request['runtime']}, {request['threads']} threads)"
    if completed.returncode:
        raise RuntimeError(
            f"{context}: worker exited {completed.returncode}\n{completed.stderr[-8000:]}"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "version": PROTOCOL_VERSION,
            "runtime": request["runtime"],
            "case": request["fixture"]["case"],
            "fixture_sha256": request["fixture"]["sha256"],
            "threads": request["threads"],
        }
        if any(result[key] != value for key, value in expected.items()):
            raise ValueError("worker identity or fixture mismatch")
        payload = result["payload"]
        if request["runtime"] == RUNTIMES[0]:
            for row in [*payload["raw"], payload["aggregate"]]:
                if any(
                    row[key] != value
                    for key, value in (
                        ("case", request["fixture"]["case"]),
                        ("threads", request["threads"]),
                        ("repeat", request["repeat"]),
                        ("warmup", request["warmup"]),
                        ("max_repeat_time", request["max_repeat_time"]),
                    )
                ):
                    raise ValueError("CPU measurement metadata mismatch")
            if any(row["runtime"] != RUNTIMES[0] for row in payload["raw"]):
                raise ValueError("CPU worker returned another runtime's measurements")
        elif (payload["error"] is not None and not isinstance(payload["error"], str)) or (
            payload["error"] and payload["durations"]
        ):
            raise ValueError("ORT worker returned inconsistent diagnostics")
        durations = (
            [row["duration_s"] for row in payload["raw"]]
            if request["runtime"] == RUNTIMES[0]
            else payload["durations"]
        )
        if not durations and not (request["runtime"] == RUNTIMES[1] and payload["error"]):
            raise ValueError("worker returned no measurements or diagnostic")
        if len(durations) > request["repeat"] or any(
            not math.isfinite(value) or value <= 0 for value in durations
        ):
            raise ValueError("worker returned invalid timing samples")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{context}: invalid worker result: {exc}") from exc
    return payload


def measure_isolated(request: dict[str, Any], *, ort_first: bool) -> tuple[list[Any], Any]:
    order = RUNTIMES[::-1] if ort_first else RUNTIMES
    measurements = {}
    for runtime in order:
        measurements[runtime] = run_worker({**request, "runtime": runtime})
    cpu = measurements[RUNTIMES[0]]
    ort = measurements[RUNTIMES[1]]
    raw, aggregate = cpu["raw"], cpu["aggregate"]
    runtime_order = ",".join(
        runtime for runtime in order if runtime == RUNTIMES[0] or not ort["error"]
    )
    aggregate["runtime_order"] = runtime_order
    aggregate["onnxruntime_error"] = ort["error"]
    for row in raw:
        row["runtime_order"] = runtime_order
    if ort["durations"]:
        metadata = {
            key: value
            for key, value in raw[0].items()
            if key not in {"runtime", "run", "duration_s"}
        }
        ort_raw = [
            {**metadata, "runtime": RUNTIMES[1], "run": index, "duration_s": duration}
            for index, duration in enumerate(ort["durations"], start=1)
        ]
        raw = ort_raw + raw if ort_first else raw + ort_raw
        aggregate.update(
            onnxruntime_samples=len(ort["durations"]),
            onnxruntime_mean_s=statistics.fmean(ort["durations"]),
            onnxruntime_median_s=statistics.median(ort["durations"]),
            speedup=statistics.median(ort["durations"]) / aggregate["median_s"],
        )
    return raw, aggregate


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sys.path[:] = request["sys_path"]
    sys.meta_path.insert(0, _PinnedImports(request["modules"]))
    result = execute_worker(request)
    (Path(request["directory"]) / "result.json").write_text(
        json.dumps(result, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

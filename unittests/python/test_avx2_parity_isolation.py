import copy
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from tools import _avx2_parity_worker as worker
from tools import benchmark_avx2_parity as benchmark


@contextmanager
def workspace():
    path = Path(f".avx2-parity-test-{uuid4().hex}").resolve()
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def fake_case():
    arrays = [
        np.array([[1, -2], [3, 4]], dtype=np.float32),
        np.array([[5, 6], [-7, 8]], dtype=np.float32),
    ]
    return SimpleNamespace(
        name="test_cpu_abs_float32_benchmark",
        model=SimpleNamespace(
            SerializeToString=lambda: b"",
            graph=SimpleNamespace(input=[SimpleNamespace(name="X")]),
        ),
        data_sets=[
            SimpleNamespace(
                inputs=[SimpleNamespace(data_type=1, shape=array.shape, raw_data=array.tobytes)]
            )
            for array in arrays
        ],
        unload=lambda: None,
    )


def request_for(path):
    return {
        "version": worker.PROTOCOL_VERSION,
        "directory": str(path),
        "fixture": worker.write_fixture(fake_case(), path),
        "modules": {},
        "sys_path": sys.path.copy(),
        "repeat": 2,
        "warmup": 1,
        "max_repeat_time": 1.0,
        "threads": 3,
    }


def cpu_result(request):
    metadata = {
        "case": request["fixture"]["case"],
        "operator": "Abs",
        "dtype": "float32",
        "repeat": request["repeat"],
        "warmup": request["warmup"],
        "threads": request["threads"],
        "processor": "test",
        "input_shapes": '[{"X": [2, 2]}, {"X": [2, 2]}]',
        "max_repeat_time": request["max_repeat_time"],
        "runtime_order": "onnx-light-cpu",
    }
    return {
        "raw": [
            {**metadata, "run": i, "runtime": "onnx-light-cpu", "duration_s": duration}
            for i, duration in enumerate([1.0, 3.0], 1)
        ],
        "aggregate": {
            **metadata,
            "samples": 2,
            "median_s": 2.0,
            "onnxruntime_samples": None,
            "onnxruntime_mean_s": None,
            "onnxruntime_median_s": None,
            "onnxruntime_error": None,
            "speedup": None,
        },
    }


def result_envelope(request, payload):
    return {
        "version": worker.PROTOCOL_VERSION,
        "runtime": request["runtime"],
        "case": request["fixture"]["case"],
        "fixture_sha256": request["fixture"]["sha256"],
        "threads": request["threads"],
        "payload": payload,
    }


def expect_error(function, message):
    try:
        function()
    except (RuntimeError, ValueError) as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_fixture_round_trip_preserves_all_datasets_and_detects_changes():
    from onnx_light_cpu._benchmark import _to_numpy

    with workspace() as path:
        request = request_for(path)
        for actual, expected in zip(
            worker._load_inputs(request), fake_case().data_sets, strict=True
        ):
            np.testing.assert_array_equal(
                _to_numpy(actual.inputs[0]), _to_numpy(expected.inputs[0])
            )
        changed = copy.deepcopy(request)
        changed["fixture"]["datasets"][0][0]["shape"] = [4]
        expect_error(lambda: worker._load_inputs(changed), "manifest checksum")
        (path / "input-0-0.bin").write_bytes(b"changed")
        expect_error(lambda: worker._load_inputs(request), "Fixture checksum mismatch")


def test_sequential_workers_share_protocol_and_report_aggregation(monkeypatch):
    for ort_first in (False, True):
        with workspace() as path:
            request = request_for(path)
            events = []

            def run(command, **kwargs):
                assert kwargs == {"capture_output": True, "text": True, "check": False}
                received = json.loads(Path(command[-1]).read_text())
                runtime = received["runtime"]
                assert received == {**request, "runtime": runtime}
                assert len(events) % 2 == 0
                events.append(("start", runtime))
                payload = (
                    cpu_result(received)
                    if runtime == "onnx-light-cpu"
                    else {"durations": [2.0, 4.0], "error": None}
                )
                (path / "result.json").write_text(json.dumps(result_envelope(received, payload)))
                events.append(("exit", runtime))
                return subprocess.CompletedProcess(command, 0, stdout="noise", stderr="")

            with monkeypatch.context() as patch:
                patch.setattr(worker.subprocess, "run", run)
                raw, aggregate = worker.measure_isolated(request, ort_first=ort_first)
            order = worker.RUNTIMES[::-1] if ort_first else worker.RUNTIMES
            assert events == [
                (event, runtime) for runtime in order for event in ("start", "exit")
            ]
            assert [row["runtime"] for row in raw] == [r for r in order for _ in range(2)]
            assert aggregate["onnxruntime_samples"] == 2
            assert aggregate["onnxruntime_mean_s"] == 3.0
            assert aggregate["speedup"] == 1.5
            for row in (*raw, aggregate):
                row["thread_policy"] = "physical"
            report = benchmark.build_report(
                raw, [aggregate], environment="shared", command="test", simd_level="AVX2"
            )
            assert report["raw"] == raw
            assert report["results"][0]["onnx_light_cpu_samples_s"] == [1.0, 3.0]
            assert report["results"][0]["onnxruntime_samples_s"] == [2.0, 4.0]
            assert report["results"][0]["runtime_order"] == list(order)
            assert report["metadata"]["compiled_source_revision"] is None
            assert "not the compiled" in report["metadata"]["git_revision_scope"]


def test_unsupported_ort_retains_cpu_samples_and_diagnostic(monkeypatch):
    with workspace() as path:
        request = request_for(path)
        calls = []

        def run(received):
            calls.append(received["runtime"])
            if received["runtime"] == "onnx-light-cpu":
                return cpu_result(received)
            return {"durations": [], "error": "Unsupported model opset"}

        monkeypatch.setattr(worker, "run_worker", run)
        raw, aggregate = worker.measure_isolated(request, ort_first=True)
        assert calls == ["onnxruntime", "onnx-light-cpu"]
        assert aggregate["onnxruntime_error"] == "Unsupported model opset"
        assert aggregate["onnxruntime_samples"] is None
        assert aggregate["runtime_order"] == "onnx-light-cpu"
        assert [row["duration_s"] for row in raw] == [1.0, 3.0]


def test_worker_crashes_and_invalid_protocol_fail_without_starting_next(monkeypatch):
    with workspace() as path:
        request = request_for(path)
        for failure in ("exit", "missing", "json", "identity", "samples", "metadata"):
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                received = json.loads(Path(command[-1]).read_text())
                result = result_envelope(received, cpu_result(received))
                if failure == "exit":
                    return subprocess.CompletedProcess(command, 7, "", "native failure")
                if failure == "identity":
                    result["fixture_sha256"] = "wrong fixture"
                if failure == "samples":
                    result["payload"]["raw"][0]["duration_s"] = -1
                if failure == "metadata":
                    result["payload"]["aggregate"]["threads"] = 99
                if failure != "missing":
                    (path / "result.json").write_text(
                        "{" if failure == "json" else json.dumps(result)
                    )
                return subprocess.CompletedProcess(command, 0, "", "")

            monkeypatch.setattr(worker.subprocess, "run", run)
            # A stale response must never rescue a failed/missing worker response.
            (path / "result.json").write_text("{}")
            expect_error(
                lambda: worker.measure_isolated(request, ort_first=False),
                "worker exited 7" if failure == "exit" else "invalid worker result",
            )
            assert len(calls) == 1


def test_ort_worker_uses_shared_feeds_and_normal_explicit_thread_settings(monkeypatch):
    import onnx_light_cpu
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail

    from onnx_light_cpu import _benchmark

    def no_cpu(*args, **kwargs):
        raise AssertionError("ORT worker must not initialize or measure a CPU evaluator")

    monkeypatch.setattr(onnx_light_cpu, "register_kernels", no_cpu)
    monkeypatch.setattr(_benchmark, "_measure_case", no_cpu)
    with workspace() as path:
        request = {**request_for(path), "runtime": "onnxruntime"}
        calls = []

        class Session:
            def __init__(self, model, sess_options, providers):
                assert model == (path / "model.onnx").read_bytes()
                assert vars(sess_options) == {
                    "intra_op_num_threads": 3,
                    "inter_op_num_threads": 1,
                    "execution_mode": "sequential",
                }
                assert providers == ["CPUExecutionProvider"]

            def run(self, outputs, feed):
                assert outputs is None
                calls.append(feed["X"].copy())

        monkeypatch.setitem(
            sys.modules,
            "onnxruntime",
            SimpleNamespace(
                SessionOptions=SimpleNamespace,
                ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                InferenceSession=Session,
            ),
        )
        result = worker.execute_worker(request)
        assert result["fixture_sha256"] == request["fixture"]["sha256"]
        assert len(result["payload"]["durations"]) == 2
        assert result["payload"]["error"] is None
        assert len(calls) == 8  # validation, warmup, and two repeats; two feeds each.
        np.testing.assert_array_equal(calls[0], calls[2])
        np.testing.assert_array_equal(calls[1], calls[3])

        def unsupported(*args, **kwargs):
            raise Fail("opset unsupported")

        sys.modules["onnxruntime"].InferenceSession = unsupported
        assert worker.execute_worker(request)["payload"] == {
            "durations": [],
            "error": "opset unsupported",
        }

        def programming_error(*args, **kwargs):
            raise ValueError("unexpected programming error")

        sys.modules["onnxruntime"].InferenceSession = programming_error
        expect_error(lambda: worker.execute_worker(request), "unexpected programming error")


def test_cpu_worker_reuses_existing_cpu_only_measurement(monkeypatch):
    import onnx_light_cpu
    from onnx_light_cpu import _benchmark

    with workspace() as path:
        request = {**request_for(path), "runtime": "onnx-light-cpu"}
        calls = []
        monkeypatch.setattr(onnx_light_cpu, "register_kernels", lambda: None)
        monkeypatch.setattr(
            onnx_light_cpu, "detect_simd_level", lambda: SimpleNamespace(name="AVX2")
        )
        monkeypatch.setattr(worker, "verify_native_modules", lambda modules: None)

        def measure(case, **kwargs):
            calls.append(kwargs)
            assert case.name == request["fixture"]["case"]
            assert case.model.SerializeToString() == b""
            assert len(case.data_sets) == 2
            np.testing.assert_array_equal(
                _benchmark._to_numpy(case.data_sets[1].inputs[0]),
                _benchmark._to_numpy(fake_case().data_sets[1].inputs[0]),
            )
            payload = cpu_result(request)
            return payload["raw"], payload["aggregate"]

        monkeypatch.setattr(_benchmark, "_measure_case", measure)
        assert worker.execute_worker(request)["payload"] == cpu_result(request)
        assert calls == [
            {
                "repeat": 2,
                "warmup": 1,
                "max_repeat_time": 1.0,
                "threads": 3,
                "with_onnxruntime": False,
            }
        ]


def test_pinned_imports_and_native_mismatch(monkeypatch):
    with workspace() as path:
        module_path = path / "pinned.py"
        module_path.write_text("value = 17\n")
        modules = {
            "parity_test_pinned": {
                "path": str(module_path),
                "package_paths": None,
                "sha256": worker.file_digest(module_path),
            }
        }
        finder = worker._PinnedImports(modules)
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        try:
            worker.verify_native_modules(modules)
            assert sys.modules["parity_test_pinned"].value == 17
            assert finder.find_spec("not_pinned") is None
            module_path.write_text("value = 18\n")
            expect_error(lambda: worker.verify_native_modules(modules), "changed on disk")
        finally:
            sys.modules.pop("parity_test_pinned", None)


def test_shared_library_mismatch_is_not_hidden_by_unchanged_extensions(monkeypatch):
    expected = {"/runtime/liblib_onnx_light_cpu.so": "original-kernel"}
    monkeypatch.setattr(worker, "shared_library_snapshot", lambda: dict(expected))
    worker.verify_shared_libraries(expected)
    monkeypatch.setattr(
        worker,
        "shared_library_snapshot",
        lambda: {"/runtime/liblib_onnx_light_cpu.so": "rebuilt-kernel"},
    )
    expect_error(lambda: worker.verify_shared_libraries(expected), "shared libraries differ")
    monkeypatch.setattr(worker, "shared_library_snapshot", lambda: None)
    expect_error(lambda: worker.verify_shared_libraries(expected), "shared libraries differ")
    worker.verify_shared_libraries(None)


def test_real_process_protocol_waits_for_exit_without_running_kernels(monkeypatch):
    """Use real Python processes with synthetic durations, not runtime benchmarks."""
    with workspace() as path:
        request = request_for(path)
        script = path / "fake_worker.py"
        script.write_text(
            "import atexit, json, os, sys\n"
            "from pathlib import Path\n"
            "request = json.loads(Path(sys.argv[1]).read_text())\n"
            "directory = Path(request['directory'])\n"
            "lock = directory / 'active'\n"
            "lock.mkdir()\n"
            "atexit.register(lock.rmdir)\n"
            "with (directory / 'pids').open('a') as f: f.write(str(os.getpid()) + '\\n')\n"
            "payload = request['fake_payloads'][request['runtime']]\n"
            "result = {key: request[key] for key in ('version', 'runtime', 'threads')}\n"
            "result.update(case=request['fixture']['case'], "
            "fixture_sha256=request['fixture']['sha256'], payload=payload)\n"
            "(directory / 'result.json').write_text(json.dumps(result))\n"
        )
        request["fake_payloads"] = {
            "onnx-light-cpu": cpu_result(request),
            "onnxruntime": {"durations": [2.0], "error": None},
        }
        monkeypatch.setattr(worker, "__file__", str(script))
        raw, aggregate = worker.measure_isolated(request, ort_first=False)
        pids = [int(pid) for pid in (path / "pids").read_text().splitlines()]
        assert len(set(pids)) == 2
        assert os.getpid() not in pids
        assert len(raw) == 3
        assert aggregate["speedup"] == 1.0


def test_native_import_snapshot_survives_an_editable_redirect_without_running_kernels():
    from onnx_light_cpu import detect_simd_level

    detect_simd_level()
    modules = worker.import_snapshot()
    native = {
        name: entry
        for name, entry in modules.items()
        if name.startswith("onnx_light_cpu.") and entry["sha256"]
    }
    assert native
    with workspace() as path:
        snapshot = path / "imports.json"
        snapshot.write_text(json.dumps(modules))
        script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from tools._avx2_parity_worker import _PinnedImports, verify_native_modules\n"
            "modules = json.loads(Path(sys.argv[1]).read_text())\n"
            "class StaleRedirect:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname in modules: raise AssertionError('stale redirect used')\n"
            "sys.meta_path[:0] = [_PinnedImports(modules), StaleRedirect()]\n"
            "import onnx_light.onnx.reference\n"
            "verify_native_modules(modules)\n"
            "assert 'onnxruntime' not in sys.modules\n"
            "print('native identities match')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(snapshot)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == "native identities match"


def test_run_selects_both_explicit_thread_policies_and_cleans_fixtures(monkeypatch):
    import onnx_light.onnx.backend as backend
    import onnx_light_cpu

    monkeypatch.setattr(
        onnx_light_cpu, "detect_simd_level", lambda: onnx_light_cpu.SimdLevel.AVX2
    )
    monkeypatch.setattr(onnx_light_cpu, "register_backend_test_cases", lambda: None)
    monkeypatch.setattr(worker, "import_snapshot", dict)
    monkeypatch.setattr(benchmark, "physical_core_count", lambda: 7)
    monkeypatch.setattr(benchmark, "AVX2_CORPUS", ((r"^test_cpu_abs_", ("float32",)),))
    calls = []
    unloaded = []
    directories = []
    case = fake_case()
    case.unload = lambda: unloaded.append(True)
    selections = []

    def collect(pattern, **kwargs):
        selections.append((pattern, kwargs))
        return [case]

    def measure(request, *, ort_first):
        assert not ort_first
        directories.append(Path(request["directory"]))
        assert directories[-1].is_dir()
        calls.append(request)
        payload = cpu_result(request)
        payload["aggregate"]["onnxruntime_error"] = "unsupported"
        return payload["raw"], payload["aggregate"]

    monkeypatch.setattr(backend, "collect_test_cases_by_name", collect)
    monkeypatch.setattr(worker, "measure_isolated", measure)
    report = benchmark.run(benchmark.parse_args(["--repeat", "2", "--dtype", "float32"]))
    assert [call["threads"] for call in calls] == [1, 7]
    assert calls[0]["fixture"] == calls[1]["fixture"]
    assert [row["thread_policy"] for row in report["unsupported"]] == ["1", "physical"]
    assert len(unloaded) == 1
    assert len(selections) == 1
    assert all(not directory.exists() for directory in directories)
    assert all(not kwargs["generate_benchmark_expected_outputs"] for _, kwargs in selections)
    assert "sequential subprocesses" in report["metadata"]["runtime_isolation"]

    def fail(request, **kwargs):
        directories.append(Path(request["directory"]))
        raise RuntimeError("worker failed")

    monkeypatch.setattr(worker, "measure_isolated", fail)
    expect_error(lambda: benchmark.run(benchmark.parse_args([])), "worker failed")
    assert len(unloaded) == 2
    assert all(not directory.exists() for directory in directories)

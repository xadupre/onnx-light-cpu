import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = """
import importlib.abc
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

class RejectPackaging(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"setuptools", "scikit_build_core"}:
            raise AssertionError(f"Native build imported packaging: {fullname}")
        return None

sys.meta_path.insert(0, RejectPackaging())
package = Path(os.environ["ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR"]) / "onnx_light"
onnx_light = types.ModuleType("onnx_light")
onnx_light.__file__ = str(package / "__init__.py")
onnx_light.get_cpp_build_info = lambda: {
    "include_dir": str(package),
    "library_dir": str(package),
    "core_library": str(package / "core.so"),
    "proto_library": str(package / "proto.so"),
}
sys.modules["onnx_light"] = onnx_light
commands = []

def record_command(command, check):
    assert check
    commands.append(command)
    if os.environ.get("FAIL_BUILD") and "--build" in command:
        raise subprocess.CalledProcessError(1, command)

sys.argv = sys.argv[1:]
with patch("subprocess.run", side_effect=record_command), patch("os.cpu_count", return_value=3):
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        print("COMMANDS=" + json.dumps(commands))
"""


class TestSetupBuildExt(unittest.TestCase):
    def run_setup(self, *args, env_overrides=None):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "onnx-light"
            package = source / "onnx_light"
            header = package / "onnx_core/runtime/kernels/kernel_dispatch_table.h"
            header.parent.mkdir(parents=True)
            header.touch()
            for filename in ("__init__.py", "core.so", "proto.so", "onnx_lightConfig.cmake"):
                (package / filename).touch()
            env = os.environ.copy()
            for name in (
                "CMAKE_ARGS",
                "CMAKE_BUILD_PARALLEL_LEVEL",
                "ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB_DIR",
                "FAIL_BUILD",
            ):
                env.pop(name, None)
            env["PYTHONPATH"] = ""
            env["ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR"] = str(source)
            env.update(env_overrides or {})
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    BOOTSTRAP,
                    str(ROOT / "setup.py"),
                    "build_ext",
                    *args,
                ],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        recorded = [line for line in result.stdout.splitlines() if line.startswith("COMMANDS=")]
        self.assertEqual(len(recorded), 1, result.stderr)
        return result, json.loads(recorded[0].removeprefix("COMMANDS="))

    def test_inplace_source_build_never_imports_packaging(self):
        result, commands = self.run_setup(
            "--inplace", "--onnx-light-source", "--cpp-tests", "--parallel", "2"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 4)
        configure, build, install, ctest = commands
        self.assertIn("-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", configure)
        self.assertIn("-DONNX_LIGHT_CPU_BUILD_TESTS=ON", configure)
        for name, filename in (
            ("ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY", "core.so"),
            ("ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY", "proto.so"),
        ):
            values = [arg for arg in configure if arg.startswith(f"-D{name}=")]
            self.assertEqual(len(values), 1)
            self.assertTrue(values[0].endswith(filename))
        self.assertEqual(build[-2:], ["--parallel", "2"])
        self.assertEqual(install[-2:], ["--prefix", str(ROOT)])
        self.assertEqual(ctest[0], "ctest")
        self.assertIn("--output-on-failure", ctest)

    def test_cmake_integration_and_short_options(self):
        result, commands = self.run_setup("-i", "--onnx-light", "-t", "native build", "-j2")
        self.assertEqual(result.returncode, 0, result.stderr)
        configure, build, install = commands
        self.assertTrue(any(arg.startswith("-Donnx_light_DIR=") for arg in configure))
        self.assertEqual(Path(build[2]).name, "native build")
        self.assertEqual(build[-2:], ["--parallel", "2"])
        self.assertEqual(install[-1], str(ROOT))

    def test_non_inplace_build_and_default_parallelism(self):
        result, commands = self.run_setup("--build-temp=native", "--build-lib=output")
        self.assertEqual(result.returncode, 0, result.stderr)
        configure, build, install = commands
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", configure)
        self.assertEqual(Path(build[2]).name, "native")
        self.assertEqual(build[-2:], ["--parallel", "3"])
        self.assertEqual(Path(install[-1]).name, "output")

    def test_environment_and_explicit_overrides(self):
        env = {
            "CMAKE_ARGS": "-DCMAKE_BUILD_TYPE=Debug -DEXAMPLE=ON",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        }
        for args in ((), ("--parallel=4",)):
            with self.subTest(args=args):
                result, commands = self.run_setup(*args, env_overrides=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("-DCMAKE_BUILD_TYPE=Debug", commands[0])
                self.assertNotIn("-DCMAKE_BUILD_TYPE=Release", commands[0])
                self.assertIn("-DEXAMPLE=ON", commands[0])
                if args:
                    self.assertEqual(commands[1][-2:], ["--parallel", "4"])
                else:
                    self.assertNotIn("--parallel", commands[1])

    def test_help_without_packaging(self):
        result, commands = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--onnx-light-source", result.stdout)
        self.assertEqual(commands, [])

    def test_invalid_arguments_do_not_start_cmake(self):
        for args in (("--parallel", "invalid"), ("--build-temp",), ("--unknown",)):
            with self.subTest(args=args):
                result, commands = self.run_setup(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertEqual(commands, [])

    def test_failed_build_does_not_install(self):
        result, commands = self.run_setup(env_overrides={"FAIL_BUILD": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("--build", commands[-1])
        self.assertIn("CalledProcessError", result.stderr)


if __name__ == "__main__":
    unittest.main()

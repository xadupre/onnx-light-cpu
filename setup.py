import argparse
import os
import shlex
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def _cmake_args_from_env():
    cmake_args = os.environ.get("CMAKE_ARGS")
    if not cmake_args:
        return []
    return shlex.split(cmake_args)


def _set_cmake_define(cmake_args, name, value):
    prefix = f"-D{name}="
    filtered = [arg for arg in cmake_args if not arg.startswith(prefix)]
    filtered.append(f"{prefix}{value}")
    return filtered


def _set_cmake_default_define(cmake_args, name, value):
    """Sets a default CMake define only when it is not already present."""
    prefix = f"-D{name}="
    if any(arg.startswith(prefix) for arg in cmake_args):
        return cmake_args
    return [*cmake_args, f"{prefix}{value}"]


def _default_parallel_jobs():
    """Returns default parallel jobs for CMake builds."""
    cmake_parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    if cmake_parallel:
        return None
    return os.cpu_count() or 1


def _ctest_command(build_temp):
    """Returns the ctest command that runs the C++ unit tests."""
    return [
        "ctest",
        "--test-dir",
        str(build_temp),
        "--output-on-failure",
        "--build-config",
        "Release",
        "--timeout",
        "240",
        "--repeat",
        "until-pass:2",
    ]


def _onnx_light_cmake_dir():
    """Returns the CMake config directory of a locally built onnx-light.

    onnx-light must be importable (``import onnx_light``) and already built so
    that its ``onnx_lightConfig.cmake`` is available for ``find_package``. The
    file is located by importing onnx-light and searching the tree that holds
    the package for ``onnx_lightConfig.cmake``.
    """
    import onnx_light

    package_dir = Path(onnx_light.__file__).resolve().parent
    search_root = package_dir.parent
    matches = sorted(search_root.glob("**/onnx_lightConfig.cmake"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find 'onnx_lightConfig.cmake' under {search_root}. Build "
            "onnx-light locally before using --onnx-light."
        )
    return matches[0].parent


def _onnx_light_from_pythonpath(header):
    """Returns the first onnx-light package explicitly selected by PYTHONPATH."""
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        package_dir = Path(entry).resolve() / "onnx_light"
        if (package_dir / header).is_file():
            return package_dir
    return None


def _onnx_light_source_build_info():
    """Returns paths for the C++ runtime built in the local onnx-light tree."""
    import onnx_light

    header = Path("onnx_core/runtime/kernels/kernel_dispatch_table.h")
    pythonpath_dir = _onnx_light_from_pythonpath(header)
    configured_source = os.environ.get("ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR")
    if pythonpath_dir is not None:
        include_dir = pythonpath_dir
    elif configured_source:
        configured_dir = Path(configured_source).resolve()
        include_dir = (
            configured_dir / "onnx_light"
            if (configured_dir / "onnx_light" / header).is_file()
            else configured_dir
        )
        if not (include_dir / header).is_file():
            raise FileNotFoundError(
                "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR does not contain the "
                f"onnx-light headers: {configured_dir}"
            )
    else:
        sibling_dir = Path(__file__).resolve().parent.parent / "onnx-light" / "onnx_light"
        if (sibling_dir / header).is_file():
            include_dir = sibling_dir
        else:
            include_dir = Path(onnx_light.__file__).resolve().parent

    if not (include_dir / header).is_file():
        raise FileNotFoundError(
            f"Could not find the onnx-light C++ headers under {include_dir}. Build "
            "onnx-light from a sibling checkout, add it to PYTHONPATH, or set "
            "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR before using --onnx-light-source."
        )

    if pythonpath_dir is not None:
        imported_dir = Path(onnx_light.__file__).resolve().parent
        if imported_dir != include_dir:
            raise RuntimeError(
                f"PYTHONPATH selects onnx-light from {include_dir}, but Python imported "
                f"it from {imported_dir}."
            )
        extension_spec = find_spec("onnx_light.onnx_py._onnxpyprotoop")
        extension_path = (
            Path(extension_spec.origin).resolve()
            if extension_spec is not None and extension_spec.origin is not None
            else None
        )
        runtime_dir = include_dir / "onnx_py"
        if extension_path is None or extension_path.parent != runtime_dir:
            raise RuntimeError(
                f"PYTHONPATH selects onnx-light from {include_dir}, but its native "
                f"extension resolves to {extension_path}. Remove the conflicting "
                "onnx-light installation; mixing runtimes is not supported."
            )

        def find_runtime_library(name):
            for pattern in (f"lib{name}.so", f"lib{name}.dylib", f"{name}.dll"):
                matches = sorted(runtime_dir.glob(pattern))
                if matches:
                    return str(matches[0].resolve())
            raise FileNotFoundError(
                f"Could not find {name} next to the PYTHONPATH runtime in {runtime_dir}."
            )

        info = {
            "include_dir": str(include_dir),
            "library_dir": str(runtime_dir),
            "core_library": find_runtime_library("lib_onnx_core"),
            "proto_library": find_runtime_library("lib_onnx_proto"),
        }
    else:
        info = dict(onnx_light.get_cpp_build_info())
        info["include_dir"] = str(include_dir)

    for key in ("core_library", "proto_library"):
        if key not in info or not Path(info[key]).is_file():
            raise FileNotFoundError(
                f"onnx-light did not report a usable {key!r}. Build and install "
                "onnx-light before using --onnx-light-source."
            )
    import_library_dir = os.environ.get("ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB_DIR")
    if import_library_dir:
        root = Path(import_library_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"onnx-light import-library directory does not exist: {root}")
        components = {
            "core_import_library": "lib_onnx_core.lib",
            "proto_import_library": "lib_onnx_proto.lib",
            "kernels_import_library": "lib_onnx_kernels.lib",
            "backend_test_import_library": "lib_onnx_backend_test.lib",
            "op_import_library": "lib_onnx_op.lib",
            "shape_import_library": "lib_onnx_shape.lib",
        }
        for key, filename in components.items():
            matches = sorted(root.glob(f"**/{filename}"))
            if not matches:
                raise FileNotFoundError(f"Could not find {filename!r} under {root}.")
            info[key] = str(matches[0].resolve())
    return info


def _add_onnx_light_defines(cmake_args):
    """Enables the onnx-light integration against a locally built onnx-light."""
    cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
    return _set_cmake_define(cmake_args, "onnx_light_DIR", str(_onnx_light_cmake_dir()))


def _add_onnx_light_source_defines(cmake_args, build_info=None):
    """Links the integration to the C++ runtime loaded by local onnx-light."""
    info = _onnx_light_source_build_info() if build_info is None else build_info
    cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
    cmake_args = _set_cmake_define(
        cmake_args,
        "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR",
        str(Path(info["include_dir"]).parent),
    )
    cmake_args = _set_cmake_define(
        cmake_args, "ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY", info["core_library"]
    )
    cmake_args = _set_cmake_define(
        cmake_args,
        "ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY",
        info["proto_library"],
    )
    if "core_import_library" in info:
        cmake_args = _set_cmake_define(
            cmake_args,
            "ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB",
            info["core_import_library"],
        )
        cmake_args = _set_cmake_define(
            cmake_args,
            "ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB",
            info["proto_import_library"],
        )
        if "kernels_import_library" in info:
            cmake_args = _set_cmake_define(
                cmake_args,
                "ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB",
                info["kernels_import_library"],
            )
        if "backend_test_import_library" in info:
            cmake_args = _set_cmake_define(
                cmake_args,
                "ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB",
                info["backend_test_import_library"],
            )
        if "op_import_library" in info:
            cmake_args = _set_cmake_define(
                cmake_args,
                "ONNX_LIGHT_CPU_ONNX_LIGHT_OP_IMPLIB",
                info["op_import_library"],
            )
        if "shape_import_library" in info:
            cmake_args = _set_cmake_define(
                cmake_args,
                "ONNX_LIGHT_CPU_ONNX_LIGHT_SHAPE_IMPLIB",
                info["shape_import_library"],
            )
    return cmake_args


def _spawn(command):
    """Prints and executes a command."""
    print(" ".join(shlex.quote(cmd_part) for cmd_part in command))
    subprocess.run(command, check=True)


def _build_extension(
    inplace=False,
    cpp_tests=False,
    onnx_light=False,
    onnx_light_source=False,
    build_temp="build/temp",
    build_lib="build/lib",
    parallel=None,
):
    """Configures, builds, and installs the extension directly with CMake."""
    root = Path(__file__).resolve().parent
    build_temp_path = Path(build_temp).resolve()
    build_temp_path.mkdir(parents=True, exist_ok=True)
    if parallel is None:
        parallel = _default_parallel_jobs()

    install_prefix = root if inplace else Path(build_lib).resolve()
    cmake_args = _cmake_args_from_env()
    cmake_args = _set_cmake_default_define(cmake_args, "CMAKE_BUILD_TYPE", "Release")
    if cpp_tests:
        cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_BUILD_TESTS", "ON")
    if onnx_light:
        cmake_args = _add_onnx_light_defines(cmake_args)
    if onnx_light_source:
        cmake_args = _add_onnx_light_source_defines(cmake_args)
    _spawn(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_temp_path),
            f"-DPython_EXECUTABLE={sys.executable}",
            *cmake_args,
        ]
    )
    build_cmd = ["cmake", "--build", str(build_temp_path), "--config", "Release"]
    if parallel is not None:
        build_cmd += ["--parallel", str(parallel)]
    _spawn(build_cmd)
    _spawn(
        [
            "cmake",
            "--install",
            str(build_temp_path),
            "--config",
            "Release",
            "--prefix",
            str(install_prefix),
        ]
    )
    if cpp_tests:
        _spawn(_ctest_command(build_temp_path))


if sys.argv[1:2] == ["build_ext"]:
    # Native builds must not load setuptools plugins, even when setuptools is installed.
    parser = argparse.ArgumentParser(prog=f"{sys.argv[0]} build_ext")
    parser.add_argument("--inplace", "-i", action="store_true")
    parser.add_argument("--cpp-tests", action="store_true")
    parser.add_argument("--onnx-light", action="store_true")
    parser.add_argument("--onnx-light-source", action="store_true")
    parser.add_argument("--build-temp", "-t", default="build/temp")
    parser.add_argument("--build-lib", "-b", default="build/lib")
    parser.add_argument("--parallel", "-j", type=int)
    options = parser.parse_args(sys.argv[2:])
    print("running build_ext")
    _build_extension(**vars(options))
    raise SystemExit(0)


from setuptools import Command, Distribution, setup  # noqa: E402


class NoConfigDistribution(Distribution):
    """Skips setup.cfg and pyproject.toml parsing for setup.py commands."""

    def parse_config_files(self, _filenames=None):
        """Skips setuptools configuration file parsing."""
        return None


class BuildExt(Command):
    """Builds the extension with CMake."""

    description = "builds C++ extension with CMake"
    user_options = [
        ("inplace", "i", "build extension in the source tree"),
        ("build-temp=", "t", "temporary build directory"),
        ("build-lib=", "b", "build directory for platform-specific files"),
        ("cpp-tests", None, "build and run the C++ unit tests"),
        (
            "onnx-light",
            None,
            (
                "build the onnx-light kernel-registration integration against a "
                "locally built, importable onnx-light"
            ),
        ),
        (
            "onnx-light-source",
            None,
            (
                "build the kernel-registration integration against the already-built "
                "C++ runtime loaded by a local, importable onnx-light"
            ),
        ),
        ("parallel=", "j", "number of parallel build jobs"),
    ]
    boolean_options = [
        "inplace",
        "cpp-tests",
        "onnx-light",
        "onnx-light-source",
    ]

    def initialize_options(self):
        """Initializes default values for command options."""
        self.inplace = False
        self.build_temp = None
        self.build_lib = None
        self.cpp_tests = False
        self.onnx_light = False
        self.onnx_light_source = False
        self.parallel = _default_parallel_jobs()

    def finalize_options(self):
        """Finalizes build directory paths for unspecified options."""
        build_base = "build"
        if self.build_temp is None:
            self.build_temp = os.path.join(build_base, "temp")
        if self.build_lib is None:
            self.build_lib = os.path.join(build_base, "lib")

    def run(self):
        """Runs CMake configure, build, and install commands."""
        _build_extension(
            inplace=self.inplace,
            cpp_tests=self.cpp_tests,
            onnx_light=self.onnx_light,
            onnx_light_source=self.onnx_light_source,
            build_temp=self.build_temp,
            build_lib=self.build_lib,
            parallel=self.parallel,
        )


setup(
    name="onnx-light-cpu",
    version="0.1.16",
    packages=["onnx_light_cpu"],
    distclass=NoConfigDistribution,
    cmdclass={"build_ext": BuildExt},
)

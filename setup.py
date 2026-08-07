import os
import shlex
import subprocess
import sys
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
            "onnx-light locally (for example 'pip install --no-build-isolation "
            "-e .' in the onnx-light checkout) before using --onnx-light."
        )
    return matches[0].parent


def _onnx_light_source_dir():
    """Returns the source directory of a locally built onnx-light.

    onnx-light must be importable (``import onnx_light``) and built from a local
    checkout so that its ``CMakeLists.txt`` is available next to the package. The
    directory is located by importing onnx-light and taking the parent of the
    package directory, which is the onnx-light source root. This assumes the same
    sources were used to build the installed onnx-light Python package.
    """
    import onnx_light

    package_dir = Path(onnx_light.__file__).resolve().parent
    source_root = package_dir.parent
    cmake_lists = source_root / "CMakeLists.txt"
    if not cmake_lists.is_file():
        raise FileNotFoundError(
            f"Could not find onnx-light's 'CMakeLists.txt' in {source_root}. Install "
            "onnx-light from a local checkout (for example 'pip install "
            "--no-build-isolation -e .' in the onnx-light source tree) before using "
            "--onnx-light-source."
        )
    return source_root


def _add_onnx_light_defines(cmake_args):
    """Enables the onnx-light integration against a locally built onnx-light."""
    cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
    return _set_cmake_define(cmake_args, "onnx_light_DIR", str(_onnx_light_cmake_dir()))


def _add_onnx_light_source_defines(cmake_args):
    """Builds the onnx-light integration from a local onnx-light source tree."""
    cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
    return _set_cmake_define(
        cmake_args, "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR", str(_onnx_light_source_dir())
    )


try:
    from setuptools import Command, Distribution, setup
except ModuleNotFoundError:

    def _spawn(command, dry_run):
        """Prints and executes a command unless dry-run mode is enabled."""
        print(" ".join(shlex.quote(cmd_part) for cmd_part in command))
        if not dry_run:
            subprocess.run(command, check=True)

    def _run_build_ext_without_packaging(args):
        """Executes build_ext without setuptools or distutils support."""
        if not args or args[0] != "build_ext":
            return False

        inplace = False
        cpp_tests = False
        onnx_light = False
        onnx_light_source = False
        dry_run = False
        build_temp = "build/temp"
        build_lib = "build/lib"
        parallel = None

        i = 1
        while i < len(args):
            arg = args[i]
            if arg in {"--inplace", "-i"}:
                inplace = True
            elif arg == "--cpp-tests":
                cpp_tests = True
            elif arg == "--onnx-light":
                onnx_light = True
            elif arg == "--onnx-light-source":
                onnx_light_source = True
            elif arg in {"--dry-run", "-n"}:
                dry_run = True
            elif arg.startswith("--build-temp="):
                build_temp = arg.split("=", 1)[1]
            elif arg.startswith("--build-lib="):
                build_lib = arg.split("=", 1)[1]
            elif arg == "--build-temp" and i + 1 < len(args):
                build_temp = args[i + 1]
                i += 1
            elif arg == "--build-lib" and i + 1 < len(args):
                build_lib = args[i + 1]
                i += 1
            elif arg.startswith("--parallel="):
                value = arg.split("=", 1)[1]
                try:
                    parallel = int(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid value for --parallel: expected an integer, got {value!r}."
                    ) from None
            elif arg in {"--parallel", "-j"} and i + 1 < len(args):
                value = args[i + 1]
                try:
                    parallel = int(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid value for --parallel: expected an integer, got {value!r}."
                    ) from None
                i += 1
            else:
                raise ValueError(f"Unsupported argument for build_ext: {arg!r}.")
            i += 1

        root = Path(__file__).resolve().parent
        build_temp_path = Path(build_temp).resolve()
        build_temp_path.mkdir(parents=True, exist_ok=True)
        if parallel is None:
            parallel = _default_parallel_jobs()

        print("running build_ext")
        install_prefix = root if inplace else Path(build_lib).resolve()
        cmake_args = _cmake_args_from_env()
        cmake_args = _set_cmake_default_define(cmake_args, "CMAKE_BUILD_TYPE", "Release")
        if cpp_tests:
            cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_BUILD_TESTS", "ON")
        if onnx_light:
            if dry_run:
                cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
            else:
                cmake_args = _add_onnx_light_defines(cmake_args)
        if onnx_light_source:
            if dry_run:
                cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
            else:
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
            ],
            dry_run,
        )
        build_cmd = ["cmake", "--build", str(build_temp_path), "--config", "Release"]
        if parallel is not None:
            build_cmd += ["--parallel", str(parallel)]
        _spawn(build_cmd, dry_run)
        _spawn(
            [
                "cmake",
                "--install",
                str(build_temp_path),
                "--prefix",
                str(install_prefix),
            ],
            dry_run,
        )
        if cpp_tests:
            _spawn(_ctest_command(build_temp_path), dry_run)
        return True

    if _run_build_ext_without_packaging(sys.argv[1:]):
        raise SystemExit(0) from None
    raise


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
            "build the onnx-light kernel-registration integration against a "
            "locally built, importable onnx-light",
        ),
        (
            "onnx-light-source",
            None,
            "build the onnx-light kernel-registration integration from a local "
            "onnx-light source tree (auto-discovered from the importable "
            "onnx-light) instead of find_package(onnx_light)",
        ),
        ("parallel=", "j", "number of parallel build jobs"),
        (
            "dry-run",
            "n",
            "print the cmake/ctest commands without executing them",
        ),
    ]
    boolean_options = ["inplace", "cpp-tests", "onnx-light", "onnx-light-source", "dry-run"]

    def initialize_options(self):
        """Initializes default values for command options."""
        self.inplace = False
        self.build_temp = None
        self.build_lib = None
        self.cpp_tests = False
        self.onnx_light = False
        self.onnx_light_source = False
        self.parallel = _default_parallel_jobs()
        # Newer setuptools/distutils no longer expose a global --dry-run
        # option (nor does Command.spawn() honor one), so --dry-run/-n is
        # declared as a plain option of this command instead and handled
        # explicitly in run() via _spawn_or_print().
        self.dry_run = False

    def finalize_options(self):
        """Finalizes build directory paths for unspecified options."""
        build_base = "build"
        if self.build_temp is None:
            self.build_temp = os.path.join(build_base, "temp")
        if self.build_lib is None:
            self.build_lib = os.path.join(build_base, "lib")

    def _spawn(self, cmd):
        """Prints a command and executes it unless --dry-run was requested.

        Command.spawn() delegates to distutils' own dry-run handling, which
        newer setuptools/distutils versions no longer wire up to a global
        --dry-run flag, so --dry-run is handled directly here instead.
        """
        print(" ".join(shlex.quote(str(part)) for part in cmd))
        if not self.dry_run:
            subprocess.run(cmd, check=True)

    def run(self):
        """Runs CMake configure, build, and install commands."""
        root = Path(__file__).resolve().parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        install_prefix = root if self.inplace else Path(self.build_lib).resolve()
        cmake_args = _cmake_args_from_env()
        cmake_args = _set_cmake_default_define(cmake_args, "CMAKE_BUILD_TYPE", "Release")
        if self.cpp_tests:
            cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_BUILD_TESTS", "ON")
        if self.onnx_light:
            cmake_args = _add_onnx_light_defines(cmake_args)
        if self.onnx_light_source:
            cmake_args = _add_onnx_light_source_defines(cmake_args)

        self._spawn(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build_temp),
                f"-DPython_EXECUTABLE={sys.executable}",
                *cmake_args,
            ]
        )
        build_cmd = ["cmake", "--build", str(build_temp), "--config", "Release"]
        if self.parallel is not None:
            build_cmd += ["--parallel", str(self.parallel)]
        self._spawn(build_cmd)
        self._spawn(["cmake", "--install", str(build_temp), "--prefix", str(install_prefix)])
        if self.cpp_tests:
            self._spawn(_ctest_command(build_temp))


setup(
    name="onnx-light-cpu",
    version="0.1.11",
    packages=["onnx_light_cpu"],
    distclass=NoConfigDistribution,
    cmdclass={"build_ext": BuildExt},
)

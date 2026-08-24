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


def _editable_install_onnx_py_dir(root):
    """Returns the ``onnx_py`` directory of a *separate* editable install of
    this project in site-packages, if any, or ``None`` otherwise.

    ``pip install -e .`` (scikit-build-core) hard-wires the compiled
    extension modules (``_cpukernels*``, ``liblib_onnx_light_cpu*``) to the
    copy it placed under site-packages at install time; unlike the pure
    Python sources, that copy is *not* redirected back to this source tree.
    If that copy exists but this in-place build is not itself running from
    it, it silently shadows the fresh in-place build for any module it
    contains (same file name resolved first), which can undefined-symbol at
    import time once the two builds drift apart. Keep it in sync instead of
    just relying on a separate ``pip install -e .`` step.
    """
    import contextlib
    import site

    site_dirs = []
    with contextlib.suppress(AttributeError):  # not available in venvs on some platforms
        site_dirs.extend(site.getsitepackages())
    with contextlib.suppress(AttributeError):
        site_dirs.append(site.getusersitepackages())

    for site_dir in site_dirs:
        candidate = Path(site_dir) / "onnx_light_cpu" / "onnx_py"
        try:
            if candidate.resolve() == (root / "onnx_light_cpu" / "onnx_py").resolve():
                continue  # this *is* the in-place tree (e.g. an inplace-only editable dir)
        except OSError:
            continue
        if candidate.is_dir():
            return candidate
    return None


def _sync_editable_install(root, install_prefix):
    """Copies freshly built binary artifacts into a stale editable install.

    Only overwrites files that already exist in the site-packages copy, so a
    plain (non ``--onnx-light-source``) editable install never gains
    dev-only artifacts (e.g. ``_cpuregister*``) it never shipped.
    """
    built_onnx_py = install_prefix / "onnx_light_cpu" / "onnx_py"
    if not built_onnx_py.is_dir():
        return
    editable_onnx_py = _editable_install_onnx_py_dir(root)
    if editable_onnx_py is None:
        return
    for built_file in built_onnx_py.iterdir():
        if built_file.suffix not in {".so", ".pyd", ".dll"}:
            continue
        target = editable_onnx_py / built_file.name
        if not target.exists():
            continue  # never introduce files a plain editable install never had
        print(f"copying {built_file} -> {target}")
        _copy_file(built_file, target)


def _copy_file(src, dst):
    """Copies ``src`` onto ``dst``, replacing it (``shutil.copy2`` mirror)."""
    import shutil

    shutil.copy2(src, dst)


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
            "onnx-light locally (for example 'pip install --no-build-isolation "
            "-e .' in the onnx-light checkout) before using --onnx-light."
        )
    return matches[0].parent


def _onnx_light_source_build_info():
    """Returns paths for the C++ runtime loaded by a local onnx-light.

    The shared library must be the exact copy loaded by onnx-light's Python
    extensions. Linking another copy would create a separate global kernel
    dispatch table, so registrations made by onnx-light-cpu would be invisible
    to ``ReferenceEvaluator``.
    """
    from onnx_light import get_cpp_build_info

    info = dict(get_cpp_build_info())
    include_dir = Path(info["include_dir"])
    if not (
        include_dir / "onnx_core" / "runtime" / "kernels" / "kernel_dispatch_table.h"
    ).is_file():
        raise FileNotFoundError(
            f"Could not find the onnx-light C++ headers under {include_dir}. Build "
            "onnx-light from a local checkout before using --onnx-light-source."
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
    return cmake_args


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
                cmake_args = _add_onnx_light_source_defines(
                    cmake_args,
                    {
                        "include_dir": "/onnx-light/include",
                        "core_library": "/onnx-light/lib/lib_onnx_core",
                        "proto_library": "/onnx-light/lib/lib_onnx_proto",
                        "core_import_library": "/onnx-light/lib/lib_onnx_core.lib",
                        "proto_import_library": "/onnx-light/lib/lib_onnx_proto.lib",
                        "kernels_import_library": "/onnx-light/lib/lib_onnx_kernels.lib",
                        "backend_test_import_library": (
                            "/onnx-light/lib/lib_onnx_backend_test.lib"
                        ),
                    },
                )
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
                "--config",
                "Release",
                "--prefix",
                str(install_prefix),
            ],
            dry_run,
        )
        if cpp_tests:
            _spawn(_ctest_command(build_temp_path), dry_run)
        if inplace and not dry_run:
            _sync_editable_install(root, install_prefix)
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
        (
            "dry-run",
            "n",
            "print the cmake/ctest commands without executing them",
        ),
    ]
    boolean_options = [
        "inplace",
        "cpp-tests",
        "onnx-light",
        "onnx-light-source",
        "dry-run",
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
            if self.dry_run:
                cmake_args = _set_cmake_define(cmake_args, "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT", "ON")
            else:
                cmake_args = _add_onnx_light_defines(cmake_args)
        if self.onnx_light_source:
            if self.dry_run:
                cmake_args = _add_onnx_light_source_defines(
                    cmake_args,
                    {
                        "include_dir": "/onnx-light/include",
                        "core_library": "/onnx-light/lib/lib_onnx_core",
                        "proto_library": "/onnx-light/lib/lib_onnx_proto",
                        "core_import_library": "/onnx-light/lib/lib_onnx_core.lib",
                        "proto_import_library": "/onnx-light/lib/lib_onnx_proto.lib",
                        "kernels_import_library": "/onnx-light/lib/lib_onnx_kernels.lib",
                        "backend_test_import_library": (
                            "/onnx-light/lib/lib_onnx_backend_test.lib"
                        ),
                    },
                )
            else:
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
        self._spawn(
            [
                "cmake",
                "--install",
                str(build_temp),
                "--config",
                "Release",
                "--prefix",
                str(install_prefix),
            ]
        )
        if self.cpp_tests:
            self._spawn(_ctest_command(build_temp))
        if self.inplace and not self.dry_run:
            _sync_editable_install(root, install_prefix)


setup(
    name="onnx-light-cpu",
    version="0.1.16",
    packages=["onnx_light_cpu"],
    distclass=NoConfigDistribution,
    cmdclass={"build_ext": BuildExt},
)

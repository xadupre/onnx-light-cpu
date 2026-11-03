import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools.onnx_light_release import (
    ONNX_LIGHT_VERSION,
    find_msvc_tool,
    parse_dumpbin_exports,
    wheel_name,
    wheel_url,
    windows_import_components,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_onnx_light_release_version():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    requirements = [Requirement(d) for d in dependencies if d.startswith("onnx-light")]
    assert len(requirements) == 1
    assert requirements[0].url is None
    assert requirements[0].marker is None
    assert str(requirements[0].specifier) == f">={ONNX_LIGHT_VERSION}"


def test_release_version_follows_dependency_metadata(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "onnx_light_release.py").write_text(
        (_ROOT / "tools" / "onnx_light_release.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["numpy>=2.1", "onnx-light>=1.2.3"]\n', encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tools.onnx_light_release import ONNX_LIGHT_VERSION, wheel_url; "
                "print(ONNX_LIGHT_VERSION); print(wheel_url('Linux', 'x86_64', (3, 12)))"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert result.stdout.splitlines() == [
        "1.2.3",
        (
            "https://github.com/xadupre/onnx-light/releases/download/1.2.3/"
            "onnx_light-1.2.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        ),
    ]


@pytest.mark.parametrize("python_version", [(3, 12), (3, 13), (3, 14)])
@pytest.mark.parametrize(
    ("system", "machine", "sys_platform"),
    [
        ("Linux", "x86_64", "linux"),
        ("Linux", "aarch64", "linux"),
        ("Windows", "AMD64", "win32"),
        ("Windows", "ARM64", "win32"),
        ("Darwin", "x86_64", "darwin"),
        ("Darwin", "arm64", "darwin"),
    ],
)
def test_project_requires_minimum_runtime_on_all_platforms(
    system, machine, sys_platform, python_version
):
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    environment = {
        "python_version": ".".join(map(str, python_version)),
        "sys_platform": sys_platform,
        "platform_machine": machine,
    }
    requirements = [Requirement(dependency) for dependency in metadata["project"]["dependencies"]]
    selected = [
        requirement
        for requirement in requirements
        if requirement.name == "onnx-light"
        and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]
    assert len(selected) == 1, (environment, selected)
    assert selected[0].url is None
    assert str(selected[0].specifier) == f">={ONNX_LIGHT_VERSION}"


def test_avx2_parity_uses_release_version():
    workflow = (_ROOT / ".github" / "workflows" / "avx2_parity.yml").read_text(encoding="utf-8")
    assert 'onnx_light_source=$(mktemp -d "${RUNNER_TEMP}/onnx-light.XXXXXX")' in workflow
    assert 'python tools/onnx_light_release.py clone "${onnx_light_source}"' in workflow
    assert 'python -m pip install -C wheel.py-api=cp312 "${onnx_light_source}"' in workflow
    assert (
        'echo "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR=${onnx_light_source}" >> "${GITHUB_ENV}"'
        in workflow
    )
    assert "../onnx-light" not in workflow


@pytest.mark.parametrize(
    "dependency_version", ["0.0.1", ONNX_LIGHT_VERSION, f"{ONNX_LIGHT_VERSION}.1"]
)
def test_installed_cmake_package_requires_compatible_onnx_light(tmp_path, dependency_version):
    template = (_ROOT / "cmake" / "onnx_light_cpuConfig.cmake.in").as_posix()
    (tmp_path / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.15)
project(CheckInstalledDependency LANGUAGES CXX)
include(CMakePackageConfigHelpers)
set(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT ON)
set(ONNX_LIGHT_CPU_ONNX_LIGHT_MIN_VERSION "{ONNX_LIGHT_VERSION}")
configure_package_config_file("{template}"
  "${{CMAKE_BINARY_DIR}}/onnx_light_cpuConfig.cmake" INSTALL_DESTINATION lib/cmake)
file(WRITE "${{CMAKE_BINARY_DIR}}/onnx_light_cpuTargets.cmake"
  "add_library(onnx_light_cpu::ml_cpu_compute INTERFACE IMPORTED)\\n")
file(WRITE "${{CMAKE_BINARY_DIR}}/dependency/onnx_lightConfig.cmake" "")
write_basic_package_version_file(
  "${{CMAKE_BINARY_DIR}}/dependency/onnx_lightConfigVersion.cmake"
  VERSION {dependency_version} COMPATIBILITY SameMajorVersion)
set(onnx_light_DIR "${{CMAKE_BINARY_DIR}}/dependency")
set(onnx_light_cpu_DIR "${{CMAKE_BINARY_DIR}}")
find_package(onnx_light_cpu CONFIG REQUIRED)
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if dependency_version != "0.0.1":
        assert result.returncode == 0, result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "onnx_light" in result.stdout
        assert ONNX_LIGHT_VERSION in result.stdout


def test_source_workflows_use_release_version():
    for name in ("ci_core.yml", "docs.yml", "cpp_standalone.yml"):
        workflow = (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "python tools/onnx_light_release.py clone " in workflow, name
        assert ONNX_LIGHT_VERSION not in workflow, name


def test_getting_started_uses_release_version():
    guide = (_ROOT / "docs" / "getting_started.rst").read_text(encoding="utf-8")
    assert "minimum onnx-light version declared in ``pyproject.toml``" in guide


def test_current_documentation_uses_release_version():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "minimum **onnx-light** version declared in `pyproject.toml`" in readme
    cli_guide = (_ROOT / "docs" / "command_line.rst").read_text(encoding="utf-8")
    assert "``onnx-light release`` job" in cli_guide


def test_release_wheel_names():
    assert wheel_name("Linux", "x86_64", (3, 12)) == (
        f"onnx_light-{ONNX_LIGHT_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    assert wheel_name("Linux", "aarch64", (3, 13)) == (
        f"onnx_light-{ONNX_LIGHT_VERSION}-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    )
    assert wheel_name("Darwin", "arm64", (3, 14)) == (
        f"onnx_light-{ONNX_LIGHT_VERSION}-cp314-cp314-macosx_13_0_universal2.whl"
    )
    assert wheel_name("Windows", "AMD64", (3, 12)) == (
        f"onnx_light-{ONNX_LIGHT_VERSION}-cp312-cp312-win_amd64.whl"
    )
    assert wheel_url("Windows", "ARM64", (3, 13)).endswith(
        f"/{ONNX_LIGHT_VERSION}/onnx_light-{ONNX_LIGHT_VERSION}-cp313-cp313-win_arm64.whl"
    )


def test_unsupported_release_platform():
    with pytest.raises(ValueError, match="Linux ppc64le"):
        wheel_name("Linux", "ppc64le", (3, 12))


def test_parse_dumpbin_exports():
    output = """
      ordinal hint RVA      name
            1    0 00011000 ?CreateKernel@@YAPEAXXZ
            2    1 00012000 PlainExport
    """
    assert parse_dumpbin_exports(output) == ["?CreateKernel@@YAPEAXXZ", "PlainExport"]


def test_windows_import_components_with_optional_threads(tmp_path):
    legacy_components = windows_import_components(tmp_path)
    assert "threads" not in legacy_components
    threading_dll = tmp_path / "lib_onnx_threads.dll"
    threading_dll.touch()
    assert windows_import_components(tmp_path) == [*legacy_components, "threads"]
    threading_dll.unlink()
    assert windows_import_components(tmp_path) == legacy_components


def test_find_msvc_tool(tmp_path):
    old_tool = (
        tmp_path / "VC" / "Tools" / "MSVC" / "14.40" / "bin" / "Hostx64" / "x64" / "dumpbin.exe"
    )
    new_tool = (
        tmp_path / "VC" / "Tools" / "MSVC" / "14.44" / "bin" / "Hostx64" / "x64" / "dumpbin.exe"
    )
    old_tool.parent.mkdir(parents=True)
    new_tool.parent.mkdir(parents=True)
    old_tool.touch()
    new_tool.touch()
    assert find_msvc_tool("dumpbin", "AMD64", tmp_path) == new_tool


def test_find_msvc_tool_cross_host(tmp_path):
    tool = tmp_path / "VC" / "Tools" / "MSVC" / "14.44" / "bin" / "Hostx64" / "arm64" / "lib.exe"
    tool.parent.mkdir(parents=True)
    tool.touch()
    assert find_msvc_tool("lib", "ARM64", tmp_path) == tool


def test_find_msvc_tool_discovers_visual_studio(tmp_path):
    program_files = tmp_path / "Program Files (x86)"
    vswhere = program_files / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vswhere.parent.mkdir(parents=True)
    vswhere.touch()
    installation = tmp_path / "Microsoft Visual Studio" / "2022" / "Enterprise"
    tool = (
        installation
        / "VC"
        / "Tools"
        / "MSVC"
        / "14.44"
        / "bin"
        / "Hostx64"
        / "x64"
        / "dumpbin.exe"
    )
    tool.parent.mkdir(parents=True)
    tool.touch()
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return str(installation)

    environment = {"PATH": "", "PROGRAMFILES(X86)": str(program_files)}
    assert find_msvc_tool("dumpbin", "AMD64", environment=environment, command_runner=run) == tool
    assert commands == [
        (
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-property",
                "installationPath",
            ],
            {"text": True},
        )
    ]

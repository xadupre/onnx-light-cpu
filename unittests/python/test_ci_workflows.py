# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the cross-repository CI ownership contract."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CORE_WORKFLOW = (_ROOT / ".github" / "workflows" / "ci_core.yml").read_text(encoding="utf-8")
_DOCS_WORKFLOW = (_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
_RELEASE_WORKFLOW = (_ROOT / ".github" / "workflows" / "build_release_wheel.yml").read_text(
    encoding="utf-8"
)
_CODECOV_CONFIG = (_ROOT / ".codecov.yml").read_text(encoding="utf-8")
_DEPENDABOT_CONFIG = (_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")


def test_documentation_build_is_linux_only():
    assert "runs-on: ubuntu-latest" in _DOCS_WORKFLOW
    assert "matrix:" not in _DOCS_WORKFLOW
    assert "sphinx-build -W -b html docs dist/html" in _DOCS_WORKFLOW
    assert "PYTHONPATH: ${{ github.workspace }}" in _DOCS_WORKFLOW


def test_documentation_does_not_replace_onnx_light_main():
    assert "git clone --depth 1 --branch main" in _DOCS_WORKFLOW
    assert ".[docs,dev]" not in _DOCS_WORKFLOW
    assert "--onnx-light-source" in _DOCS_WORKFLOW
    assert "--cpp-tests" not in _DOCS_WORKFLOW
    assert "python -m pytest" not in _DOCS_WORKFLOW


def test_source_integration_uses_matching_nanobind_abi():
    cmake = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert (
        "option(ONNX_LIGHT_CPU_PYTHON_STABLE_ABI\n"
        '       "Build Python extensions with CPython\'s stable ABI." ON)'
    ) in cmake
    assert "list(APPEND _onnx_light_cpu_nanobind_abi STABLE_ABI)" in cmake
    modules = {
        "_cpukernels": "PY_KERNELS_SOURCES",
        "_cpuregister": "PY_REGISTER_SOURCES",
    }
    for module, sources in modules.items():
        assert (
            f"nanobind_add_module({module} ${{_onnx_light_cpu_nanobind_abi}} ${{{sources}}})"
        ) in cmake
        assert f"nanobind_add_module({module} STABLE_ABI " not in cmake
    assert "-C wheel.py-api=cp312" in _DOCS_WORKFLOW
    assert "-C wheel.py-api=cp312" in _CORE_WORKFLOW
    assert "-C cmake.define.ONNX_LIGHT_PYTHON_STABLE_ABI=OFF" in _CORE_WORKFLOW
    assert "-DONNX_LIGHT_CPU_PYTHON_STABLE_ABI=${{ matrix.stable-abi }}" in _CORE_WORKFLOW


def test_release_wheels_build_and_test_session_registration_before_upload():
    for job in ("build_wheels_linux", "build_wheels_windows", "build_wheels_macos"):
        body = _RELEASE_WORKFLOW.split(f"  {job}:", 1)[1].split("  build_wheels_", 1)[0]
        assert 'CIBW_BUILD_FRONTEND: "pip; args: --no-build-isolation"' in body
        clone_release = (
            "python {project}/tools/onnx_light_release.py clone {project}/../onnx-light"
        )
        assert clone_release in body
        install_release = "python {project}/tools/onnx_light_release.py install"
        generate_import_libraries = (
            "python {project}/tools/onnx_light_release.py generate-import-libraries "
            "{project}/../onnx-light/build"
        )
        if job == "build_wheels_windows":
            assert generate_import_libraries in body
            assert body.index(install_release) < body.index(generate_import_libraries)
        else:
            assert generate_import_libraries not in body
        assert "CIBW_BEFORE_BUILD: >-" in body
        assert (
            "CIBW_BEFORE_BUILD: >-\n"
            "            python -m pip install scikit-build-core nanobind==3.1.0 &&\n"
            f"            {install_release}"
        ) in body
        assert f"CIBW_BEFORE_TEST: {install_release}" in body
        assert "python -m pip install -C wheel.py-api=cp312" not in body
        assert "git clone --depth 1 --branch main" not in body
        assert "-DONNX_LIGHT_CPU_RELEASE_WHEEL=ON" in body
        assert "-DONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF" in body
        assert (
            "CIBW_TEST_COMMAND: python {project}/unittests/python/wheel_registration_smoke.py"
            in body
        )
        if job == "build_wheels_macos":
            assert (
                "CIBW_REPAIR_WHEEL_COMMAND_MACOS: >-\n"
                "            DYLD_LIBRARY_PATH=\"$(python -c 'from onnx_light import "
                "get_cpp_build_info as i;\n"
                '            print(i()["library_dir"])\')" delocate-wheel --exclude '
                "liblib_onnx_\n"
                "            --require-archs x86_64,arm64 -w {dest_dir} {wheel}"
            ) in body
        else:
            assert "CIBW_REPAIR_WHEEL_COMMAND_MACOS" not in body
        assert body.index("CIBW_TEST_COMMAND:") < body.index("Attach wheels to GitHub Release")


def test_release_wheel_maps_the_core_windows_import_library():
    cmake = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    release_block = cmake.split("if(ONNX_LIGHT_CPU_RELEASE_WHEEL)", 1)[1].split(
        "option(ONNX_LIGHT_CPU_ENABLE_COVERAGE", 1
    )[0]
    assert 'if(_component STREQUAL "CORE")' in release_block
    assert 'set(ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB "${_import_lib_path}")' in release_block


def test_onnx_light_main_integration_runs_on_every_supported_os():
    source_job = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)[1].split(
        "  report_pr_benchmark:", 1
    )[0]
    for os_name in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert f"os: {os_name}" in source_job
    assert source_job.count("python-abi: stable") == 3
    assert source_job.count("python-abi: native") == 1
    assert source_job.count("matrix.python-abi == 'stable'") == 3
    assert "git clone --depth 1 --branch main" in source_job
    assert "scikit-build-core setuptools" in source_job
    assert "ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB_DIR=" in source_job
    assert "--cpp-tests --onnx-light-source" in source_job
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON" in source_job
    assert "sccache --start-server || :" in source_job
    assert "0.1.19" not in source_job


def test_cpp_coverage_is_carried_forward_between_weekly_runs():
    assert "cpp:\n    carryforward: true" in _CODECOV_CONFIG


def test_dependabot_updates_actions_and_python_dependencies():
    assert _DEPENDABOT_CONFIG.startswith("version: 2\nupdates:\n")
    for ecosystem in ("github-actions", "pip"):
        assert (
            f'  - package-ecosystem: "{ecosystem}"\n'
            '    directory: "/"\n'
            "    schedule:\n"
            '      interval: "weekly"\n'
        ) in _DEPENDABOT_CONFIG


def test_native_kernel_tests_run_in_source_integration_not_standalone():
    standalone_job, source_jobs = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)
    source_job = source_jobs.split("  report_pr_benchmark:", 1)[0]
    for filename in (
        "test_kernels_e2e.py",
        "test_kernels_doc_e2e.py",
        "test_nonzero.py",
        "test_reduce_sum_kernel.py",
        "test_scatter_nd.py",
        "test_simplified_layer_normalization.py",
        "test_skip_simplified_layer_normalization.py",
        "test_matmul_nbits_parity_benchmark.py",
        "test_tanh.py",
    ):
        exclusion = f"--ignore=unittests/python/{filename}"
        assert exclusion in standalone_job
        assert exclusion not in source_job
    assert "run: python -m pytest unittests" in source_job


def test_standalone_unix_install_retries_transient_wheel_download_failure():
    standalone_job = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)[0]
    unix_install = standalone_job.split("      - name: Build and install package (Unix)", 1)[
        1
    ].split("      - name: Build and install package (Windows)", 1)[0]
    assert "for attempt in 1 2 3; do" in unix_install
    assert "pip install" in unix_install
    assert 'if [ "$attempt" -eq 3 ]; then' in unix_install
    assert "exit 1" in unix_install


def test_pr_benchmark_infers_filters_and_updates_comment():
    source_job = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)[1].split(
        "  report_pr_benchmark:", 1
    )[0]
    report_job = _CORE_WORKFLOW.split("  report_pr_benchmark:", 1)[1].split(
        "  arm_gemm_native:", 1
    )[0]
    assert "runner.os == 'Linux'" in source_job
    assert "onnx_light_cpu/(backend_test/cases|impl|kernels)/" in source_job
    assert '--from-pr "${{ github.event.pull_request.html_url }}"' in source_job
    assert "actions/upload-artifact@v4" in source_job
    assert "needs: setup_onnx_light_source" in report_job
    assert "pull-requests: write" in report_job
    assert "gh pr comment" in report_job


def test_python_test_classes_inherit_from_ext_test_case():
    missing_bases = []
    for path in (_ROOT / "unittests" / "python").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                if not any(
                    (isinstance(base, ast.Name) and base.id == "ExtTestCase")
                    or (isinstance(base, ast.Attribute) and base.attr == "ExtTestCase")
                    for base in node.bases
                ):
                    missing_bases.append(f"{path.relative_to(_ROOT)}:{node.name}")
    assert not missing_bases, f"Test classes must inherit from ExtTestCase: {missing_bases}"

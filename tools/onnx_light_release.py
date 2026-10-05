import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ONNX_LIGHT_VERSION = "0.1.30"


def wheel_name(system, machine, python_version):
    python_tag = f"cp{python_version[0]}{python_version[1]}"
    normalized_machine = machine.lower()
    if system == "Linux" and normalized_machine in {"x86_64", "aarch64"}:
        platform_tag = f"manylinux_2_27_{normalized_machine}.manylinux_2_28_{normalized_machine}"
    elif system == "Darwin":
        platform_tag = "macosx_13_0_universal2"
    elif system == "Windows" and normalized_machine in {"amd64", "arm64"}:
        platform_tag = f"win_{normalized_machine}"
    else:
        raise ValueError(f"Unsupported onnx-light release platform: {system} {machine}")
    return f"onnx_light-{ONNX_LIGHT_VERSION}-{python_tag}-{python_tag}-{platform_tag}.whl"


def wheel_url(system, machine, python_version):
    name = wheel_name(system, machine, python_version)
    return f"https://github.com/xadupre/onnx-light/releases/download/{ONNX_LIGHT_VERSION}/{name}"


def clone_source(destination):
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ONNX_LIGHT_VERSION,
            "https://github.com/xadupre/onnx-light.git",
            str(destination),
        ]
    )


def install_wheel():
    url = wheel_url(platform.system(), platform.machine(), sys.version_info[:2])
    subprocess.check_call([sys.executable, "-m", "pip", "install", url])


def parse_dumpbin_exports(output):
    pattern = re.compile(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", re.MULTILINE)
    return pattern.findall(output)


def find_msvc_tool(name, machine, installation=None, environment=None, command_runner=None):
    target = {"amd64": "x64", "arm64": "arm64"}.get(machine.lower())
    if target is None:
        raise ValueError(f"Unsupported MSVC target machine: {machine}")
    if installation is None:
        environment = os.environ if environment is None else environment
        executable = shutil.which(name, path=environment.get("PATH"))
        if executable:
            return Path(executable)
        program_files = environment.get("PROGRAMFILES(X86)")
        if not program_files:
            raise FileNotFoundError("ProgramFiles(x86) is not defined.")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        runner = subprocess.check_output if command_runner is None else command_runner
        output = runner(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-property",
                "installationPath",
            ],
            text=True,
        )
        if not output.strip():
            raise FileNotFoundError("vswhere did not find a Visual Studio installation.")
        installation = Path(output.strip())

    tools_root = Path(installation) / "VC" / "Tools" / "MSVC"
    host = f"Host{target}"
    for version in sorted(tools_root.iterdir(), reverse=True):
        preferred = version / "bin" / host / target / f"{name}.exe"
        if preferred.is_file():
            return preferred
        candidates = sorted((version / "bin").glob(f"Host*/{target}/{name}.exe"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find {name}.exe for {machine} under {tools_root}.")


def generate_import_libraries(destination):
    from onnx_light import get_cpp_build_info

    machine = platform.machine().lower()
    if platform.system() != "Windows" or machine not in {"amd64", "arm64"}:
        raise ValueError(
            f"Unsupported import-library platform: {platform.system()} {platform.machine()}"
        )
    dumpbin = find_msvc_tool("dumpbin", machine)
    librarian = find_msvc_tool("lib", machine)
    library_dir = Path(get_cpp_build_info()["library_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for component in ("core", "proto", "kernels", "backend_test", "op", "shape"):
        dll = library_dir / f"lib_onnx_{component}.dll"
        output = subprocess.check_output(
            [str(dumpbin), "/nologo", "/exports", str(dll)],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        exports = parse_dumpbin_exports(output)
        if not exports:
            raise RuntimeError(f"No exports found in {dll}.")
        definition = destination / f"lib_onnx_{component}.def"
        definition.write_text(
            f"LIBRARY {dll.name}\nEXPORTS\n" + "".join(f"  {symbol}\n" for symbol in exports),
            encoding="utf-8",
        )
        subprocess.check_call(
            [
                str(librarian),
                "/nologo",
                f"/def:{definition}",
                f"/out:{destination / f'lib_onnx_{component}.lib'}",
                f"/machine:{'X64' if machine == 'amd64' else 'ARM64'}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Use the onnx-light release compatible with onnx-light-cpu."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    clone = subparsers.add_parser("clone")
    clone.add_argument("destination", type=Path)
    subparsers.add_parser("install")
    import_libraries = subparsers.add_parser("generate-import-libraries")
    import_libraries.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.command == "clone":
        clone_source(args.destination)
    elif args.command == "install":
        install_wheel()
    else:
        generate_import_libraries(args.destination)


if __name__ == "__main__":
    main()

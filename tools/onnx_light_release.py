import argparse
import platform
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


def main():
    parser = argparse.ArgumentParser(
        description="Use the onnx-light release compatible with onnx-light-cpu."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    clone = subparsers.add_parser("clone")
    clone.add_argument("destination", type=Path)
    subparsers.add_parser("install")
    args = parser.parse_args()

    if args.command == "clone":
        clone_source(args.destination)
    else:
        install_wheel()


if __name__ == "__main__":
    main()

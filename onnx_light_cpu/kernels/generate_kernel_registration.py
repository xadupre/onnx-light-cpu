# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the onnx-light-cpu kernel registration aggregators.

``onnx-light-cpu`` installs its kernels into onnx-light's shared
``KernelDispatchTable`` through onnx-light's own
:cpp:func:`onnx_light::core::runtime::RegisterKernelFn` -- there is a single
registration system and this script does not add another one. It only removes
the two hand-maintained *lists* that used to duplicate what the kernel sources
already express:

* ``RegisterAllKernels()`` (declared in ``register_kernels.h``) -- calls every
  per-operator ``Register<Name>Kernel()`` installer.
* ``RegisteredKernelNames()`` (declared in ``kernel_usage.h``) -- the
  ``{op_type, kernel name}`` table.

Both are derived here from the kernel sources themselves so that adding a kernel
only requires writing its own ``.cc``/``.h`` (with a ``Register<Name>Kernel()``
installer that calls ``RegisterKernelFn`` and a ``kName`` member): the lists
regenerate automatically at build time.

The script mirrors onnx-light's ``_generate_*.py`` convention: it emits a C++
source file carrying an "auto-generated, do not edit" banner.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# ``void Register<Name>Kernel();`` (or ``...Kernels();``) installer declaration,
# as found in a kernel header. These are the entry points RegisterAllKernels
# must call.
_INSTALLER_DECL_RE = re.compile(r"\bvoid\s+(Register\w+)\s*\(\s*\)\s*;")

# ``void Register<Name>() { ... }`` installer definition, as found in a kernel
# ``.cc``. Captures the function name; the body is scanned separately.
_INSTALLER_DEF_RE = re.compile(r"\bvoid\s+(Register\w+)\s*\(\s*\)\s*\{")

# ``std::make_unique<KernelClass>`` inside an installer body: identifies the
# kernel class backing the immediately following RegisterKernelFn call.
_MAKE_UNIQUE_RE = re.compile(r"make_unique<\s*([A-Za-z_]\w*)\s*>")

# ``RegisterKernelFn("domain", "OpType", ...)`` call: captures the op_type.
_REGISTER_FN_RE = re.compile(r"RegisterKernelFn\(\s*\"[^\"]*\"\s*,\s*\"([^\"]+)\"")

# ``class KernelClass : ...`` / ``class KernelClass final {`` definition in a
# header. Requires the class name to be followed by class-definition syntax
# (``:`` base list, ``{`` body, or ``final``) so the word "class" appearing in
# prose/comments is not mistaken for a declaration.
_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)\b\s*(?=[:{]|\bfinal\b)")

# ``static constexpr const char *kName = "...";`` inside a kernel class.
_KNAME_RE = re.compile(r"\bkName\s*=\s*\"([^\"]+)\"")

# Comment stripping so keywords used in prose (e.g. ``// this class Foo ...``)
# never feed the class/kName regexes above.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(line: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", line))


class GeneratorError(RuntimeError):
    """Raised when the kernel sources cannot be parsed unambiguously."""


def _iter_sources(kernels_dir: pathlib.Path, suffix: str) -> list[pathlib.Path]:
    return sorted(p for p in kernels_dir.rglob(f"*{suffix}") if p.is_file())


def _split_top_level_body(text: str, open_index: int) -> str:
    """Returns the balanced ``{...}`` body starting at ``open_index``."""
    depth = 0
    for i in range(open_index, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_index : i + 1]
    raise GeneratorError("unbalanced braces while scanning an installer body")


def collect_class_names(headers: list[pathlib.Path]) -> dict[str, str]:
    """Maps every kernel class to its ``kName`` value, scanning the headers."""
    class_to_name: dict[str, str] = {}
    for header in headers:
        text = header.read_text(encoding="utf-8")
        current_class: str | None = None
        for line in text.splitlines():
            code = _strip_comments(line)
            class_match = _CLASS_RE.search(code)
            if class_match:
                current_class = class_match.group(1)
            name_match = _KNAME_RE.search(code)
            if name_match:
                if current_class is None:
                    raise GeneratorError(f"{header}: found a kName without an enclosing class")
                class_to_name[current_class] = name_match.group(1)
    return class_to_name


def collect_installer_headers(headers: list[pathlib.Path]) -> dict[str, pathlib.Path]:
    """Maps every ``Register<Name>Kernel()`` installer to its header."""
    installer_to_header: dict[str, pathlib.Path] = {}
    for header in headers:
        text = header.read_text(encoding="utf-8")
        for match in _INSTALLER_DECL_RE.finditer(text):
            installer_to_header[match.group(1)] = header
    return installer_to_header


def collect_installers(
    sources: list[pathlib.Path], class_to_name: dict[str, str]
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Returns ``[(installer, [(op_type, kName), ...]), ...]`` from the ``.cc``.

    Each installer body is scanned for ``make_unique<Class>`` / RegisterKernelFn
    pairs, in source order, so a single installer can register several ops.
    """
    installers: list[tuple[str, list[tuple[str, str]]]] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for def_match in _INSTALLER_DEF_RE.finditer(text):
            body = _strip_comments(_split_top_level_body(text, def_match.end() - 1))
            ops: list[tuple[str, str]] = []
            pending_class: str | None = None
            # Walk tokens in order so each RegisterKernelFn is paired with the
            # class of the most recent make_unique before it.
            for token in re.finditer(
                r"make_unique<\s*([A-Za-z_]\w*)\s*>|"
                r"RegisterKernelFn\(\s*\"[^\"]*\"\s*,\s*\"([^\"]+)\"",
                body,
            ):
                if token.group(1) is not None:
                    pending_class = token.group(1)
                    continue
                op_type = token.group(2)
                if pending_class is None:
                    raise GeneratorError(
                        f"{source}: RegisterKernelFn for {op_type!r} has no "
                        "preceding make_unique<KernelClass>"
                    )
                if pending_class not in class_to_name:
                    raise GeneratorError(f"{source}: kernel class {pending_class!r} has no kName")
                ops.append((op_type, class_to_name[pending_class]))
                pending_class = None
            if ops:
                installers.append((def_match.group(1), ops))
    return installers


def render(
    installer_names: list[str],
    op_pairs: list[tuple[str, str]],
    includes: list[str],
) -> str:
    include_lines = "\n".join(f'#include "{path}"' for path in includes)
    call_lines = "\n".join(f"  {name}();" for name in installer_names)
    pair_lines = "\n".join(f'      {{"{op}", "{name}"}},' for op, name in op_pairs)
    return f"""// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// AUTO-GENERATED FILE -- DO NOT EDIT BY HAND.
//
// Generated by ``onnx_light_cpu/kernels/generate_kernel_registration.py`` from
// the kernel sources under ``onnx_light_cpu/kernels/``. The build regenerates
// it automatically; run the script directly to refresh it outside the build.
//
// It defines the two aggregators that used to be hand-maintained lists:
//   * RegisterAllKernels()   -- calls every Register<Name>Kernel() installer.
//   * RegisteredKernelNames() -- the {{op_type, kName}} table.
// Both are derived from the per-operator ``RegisterKernelFn`` registrations and
// ``kName`` members in the kernel sources, so adding a kernel needs no edit
// here.

#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

{include_lines}

#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {{

void RegisterAllKernels() {{
{call_lines}
}}

const std::vector<std::pair<std::string, std::string>> &RegisteredKernelNames() {{
  static const std::vector<std::pair<std::string, std::string>> names = {{
{pair_lines}
  }};
  return names;
}}

}} // namespace onnx_light_cpu
"""


def generate(kernels_dir: pathlib.Path, repo_root: pathlib.Path) -> str:
    headers = _iter_sources(kernels_dir, ".h")
    sources = _iter_sources(kernels_dir, ".cc")

    class_to_name = collect_class_names(headers)
    installer_headers = collect_installer_headers(headers)
    installers = collect_installers(sources, class_to_name)

    if not installers:
        raise GeneratorError(f"no Register<Name>Kernel() installers found under {kernels_dir}")

    installer_names = sorted(name for name, _ in installers)

    missing = [name for name in installer_names if name not in installer_headers]
    if missing:
        raise GeneratorError(
            "installer(s) defined without a header declaration: " + ", ".join(sorted(missing))
        )

    op_pairs: list[tuple[str, str]] = []
    seen_ops: set[str] = set()
    for _, ops in installers:
        for op_type, name in ops:
            if op_type in seen_ops:
                raise GeneratorError(f"duplicate op_type registration: {op_type!r}")
            seen_ops.add(op_type)
            op_pairs.append((op_type, name))
    op_pairs.sort(key=lambda pair: pair[0])

    include_paths = sorted(
        {installer_headers[name].relative_to(repo_root).as_posix() for name in installer_names}
    )

    return render(installer_names, op_pairs, include_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the kernel .cc/.h sources.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        required=True,
        help="Repository root used to compute #include paths.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path of the C++ file to (re)write.",
    )
    args = parser.parse_args(argv)

    content = generate(args.kernels_dir.resolve(), args.repo_root.resolve())

    # Avoid rewriting (and so retriggering downstream builds) when unchanged.
    if args.output.exists() and args.output.read_text(encoding="utf-8") == content:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

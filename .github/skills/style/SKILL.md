---
name: style
description: Apply and validate onnx-light-cpu formatting and Python dependency conventions.
---

# Apply repository style

Use this skill before committing changes. Apply only the checks relevant to the changed
files, then inspect the resulting diff.

1. Format changed C++ files with the repository `.clang-format` configuration and check
   them with `clang-format --dry-run --Werror`.
2. For Python changes, run `ruff check .`, `ruff format .`, and `black .`; verify with
   `ruff format --check .` and `black --check .`.
3. Do not import the `onnx`, `mock`, or `pytest` modules in Python scripts. Use the
   repository APIs and the standard library instead. `pytest` may be used in GitHub
   Actions YAML only to invoke the test runner.
4. Run the focused tests for the changed area and review the final diff for unrelated
   formatting changes.

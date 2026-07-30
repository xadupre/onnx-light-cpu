# Copilot instructions

## C++ formatting (clang-format)

C++ sources are formatted with `clang-format` using the repository's
[`.clang-format`](../.clang-format) (LLVM style, 100-column limit). The
[`clang-format`](workflows/clang_format.yml) workflow fails the build if any
`.cc`, `.h`, or `.hpp` file is not formatted.

Format the code in place before committing:

```bash
find . \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) \
  | grep -v ".git" \
  | xargs clang-format -i
```

Or check without modifying files (mirrors CI):

```bash
find . \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) \
  | grep -v ".git" \
  | xargs clang-format --dry-run --Werror
```

## Python formatting (ruff)

Python code is checked with `ruff` (see the `Style` workflow):

```bash
ruff check . && ruff format .
```

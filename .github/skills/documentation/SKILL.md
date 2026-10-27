---
name: documentation
description: Build and validate onnx-light-cpu documentation with strict Sphinx warnings.
---

# Build and validate the documentation

Use this skill when changing reStructuredText pages, documentation examples, or generated API
documentation.

1. Inspect `docs/conf.py` and `.github/workflows/docs.yml` for extensions, generated inputs, and
   the environment required by the documentation build. Keep the docs build aligned with that
   workflow, including Doxygen and the onnx-light integration when needed.
2. Ensure each reStructuredText section title has an underline made of one repeated character
   whose length is at least the title's length. Prefer checking this through Sphinx rather than
   adding a separate unit test for documentation formatting.
3. Build from the repository root with warnings treated as errors:

   ```bash
   PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" sphinx-build -W -b html docs dist/html
   ```

4. Fix the source of every warning or build error; do not disable strict warnings to make the build
   pass. Review the final documentation diff for unrelated changes.

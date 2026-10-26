---
name: build-project
description: Build onnx-light-cpu with 48 parallel jobs, reusing the existing build tree.
---

# Build with 48 parallel jobs

Use this skill whenever compiling this repository.

1. Reuse the existing configured build tree and preserve its dependency paths and build options.
2. Build the targets needed for the task with an explicit 48-job limit:

   ```bash
   cmake --build build/temp --parallel 48 --target test_gemm_kernel test_gemm_plan
   ```

   Replace the build path and targets with the existing build tree and relevant targets.
   Do not use a smaller fixed limit, an unbounded `--parallel`, or `$(nproc)` unless
   the user explicitly requests a different limit.
3. Run the focused tests after compilation. Build parallelism does not set runtime or
   benchmark thread counts; preserve those independently.

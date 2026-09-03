<p align="center">
  <img src="docs/_static/logo.svg" alt="onnx-light-cpu logo" width="116" height="80">
</p>

# onnx-light-cpu

[![ci-core](https://github.com/xadupre/onnx-light-cpu/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/onnx-light-cpu/actions/workflows/ci_core.yml)
[![Style](https://github.com/xadupre/onnx-light-cpu/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/onnx-light-cpu/actions/workflows/style.yml)
[![Typing](https://github.com/xadupre/onnx-light-cpu/actions/workflows/typing.yml/badge.svg)](https://github.com/xadupre/onnx-light-cpu/actions/workflows/typing.yml)
[![clang-format](https://github.com/xadupre/onnx-light-cpu/actions/workflows/clang_format.yml/badge.svg)](https://github.com/xadupre/onnx-light-cpu/actions/workflows/clang_format.yml)
[![OpenSSF Scorecard](https://github.com/xadupre/onnx-light-cpu/actions/workflows/cq_scorecard.yml/badge.svg)](https://github.com/xadupre/onnx-light-cpu/actions/workflows/cq_scorecard.yml)
[![Coverage](https://codecov.io/gh/xadupre/onnx-light-cpu/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light-cpu)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/xadupre/onnx-light-cpu/blob/main/LICENSE)

Highly optimized CPU kernels for
[onnx-light](https://github.com/xadupre/onnx-light).

Implements ONNX operators with SIMD-accelerated kernels that dispatch at runtime
to the best available instruction set (AVX-512, AVX2, AVX, SSE2, or scalar
fallback).

## Build from source

### Prerequisites

- C++20 compiler with AVX2 support (GCC ≥ 11, Clang ≥ 14, MSVC ≥ 2022)
- CMake ≥ 3.15
- Python ≥ 3.12
- [nanobind](https://github.com/wjakob/nanobind) ≥ 3.0

### Python wheel (recommended)

```bash
pip install .
```

### Pixi environment

```bash
pixi install
pixi run install
pixi run test-python
```

### setup.py with C++ tests

Build the extension and run the C++ unit tests with `ctest`:

```bash
python setup.py build_ext --inplace --cpp-tests
```

### setup.py with the onnx-light integration

Build the onnx-light kernel-registration integration against a locally built,
importable onnx-light (see [onnx-light](https://github.com/xadupre/onnx-light)).
onnx-light must be built and importable (`import onnx_light`); the build locates
its `onnx_lightConfig.cmake` automatically:

```bash
python setup.py build_ext --inplace --onnx-light
```

When onnx-light was installed from a local checkout (for example
`pip install --no-build-isolation -e .` in the onnx-light source tree) but its
`onnx_lightConfig.cmake` is not available, build the integration directly from
those sources instead. `--onnx-light-source` auto-discovers the onnx-light
source tree from the importable onnx-light and compiles it with
`add_subdirectory`:

```bash
python setup.py build_ext --inplace --onnx-light-source
```

### Pure CMake (C++ only)

```bash
cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_TESTS=ON \
      -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build
```

`onnx-light-cpu` does not own a thread pool. Direct standalone kernel calls run
on the calling thread. Kernels registered with `onnx-light` use the session
`CpuExecutor` already installed by `onnx-light` itself, so thread count,
affinity, spin policy, nesting, and inspection all come from the session
execution policy.

### AVX-512 support

To enable AVX-512 codepaths (compiled and usable on AVX-512 CPUs):

```bash
cmake -S . -B build -DONNX_LIGHT_CPU_SIMD_FLAGS="-mavx512f" \
      -DONNX_LIGHT_CPU_BUILD_TESTS=ON -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
cmake --build build
```

## C++ usage

```cpp
#include <onnx_light_cpu/impl/math/math_kernels.h>

int main() {
    float input[] = {-1.0f, 2.0f, -3.0f, 4.0f};
    float output[4];
    onnx_light_cpu::AbsFloat32(input, output, 4);
    // output = {1.0f, 2.0f, 3.0f, 4.0f}
}
```

Link against `onnx_light_cpu::lib_onnx_light_cpu`:

```cmake
find_package(onnx_light_cpu REQUIRED)
target_link_libraries(my_app PRIVATE onnx_light_cpu::lib_onnx_light_cpu)
```

## Python usage

The Python extension exposes only the SIMD-detection helpers; the kernels
themselves are reached through onnx-light's runtime after registration (see
below), not as standalone numpy-like functions.

```python
from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level, has_cpu_kernels

# Check that the CPU kernel extension is available and which SIMD level it uses
assert has_cpu_kernels()
level = detect_simd_level()  # 0=None, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX512
print(f"SIMD level: {level}")
```

### Running an ONNX model with onnx-light

`register_kernels` installs the optimized kernels into
[onnx-light](https://github.com/xadupre/onnx-light)'s shared C++ kernel
dispatch table. This includes the portable `com.microsoft::CDist` and
`com.microsoft::BiasGelu` kernels, the standard numeric normalization family
(`BatchNormalization` through `RMSNormalization`), and their related runtime
metadata:

`BatchNormalization` implements the opset-15 inference and training contracts;
training produces `Y`, `running_mean`, and `running_var`.

```python
import numpy as np
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    MicrosoftKernelImplementation,
    register_kernel_for_session,
    register_kernel_global,
    register_kernels_global,
)

register_kernels_global()  # installs every kernel process-wide
register_kernel_global("", "Abs")  # installs one selected kernel process-wide
# Explicit scalar correctness-oracle family for com.microsoft:
register_kernels_global(microsoft_implementation=MicrosoftKernelImplementation.NAIVE)
sess = ReferenceEvaluator(model)  # any model containing an Abs node
register_kernel_for_session(sess, "", "Abs")  # affects only sess
(y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.0], dtype=np.float32)})
```

`register_kernels` is only available in builds compiled with the onnx-light
integration (`-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`); it wraps the compiled
`onnx_light_cpu.onnx_py._cpuregister.register_all_kernels()` binding.

### Benchmarking backend test cases

The command line can benchmark selected `TestMode.BENCHMARK` cases and write
both individual measurements and per-case statistics to the `raw` and
`aggregated` sheets of an Excel workbook:

```bash
onnx-light-cpu benchmark \
    --tests "^test_cpu_(abs|gemm)_" \
    --dtypes float32 float64 \
    --output benchmark.xlsx
```

Use `--dtypes all` to include every available dtype. The same command is
available as `python -m onnx_light_cpu benchmark`.

Global registrations are owned by the process-wide dispatch table and are
observed by sessions that resolve their nodes afterwards. Session registrations
are owned by one evaluator and do not modify another evaluator or the global
table. Repeating either operation replaces the same kernel by default; pass
`replace=False` to retain an existing registration.

Custom-domain graph construction and differentiation use the matching schema
and gradient helpers:

```python
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light_cpu import operator_schema_lookup, register_custom_gradients

builder = GraphBuilder("custom", schema_lookup=operator_schema_lookup)
gradient_registry = register_custom_gradients()
```

> **Registration seems ignored?** The kernels only take effect when
> `onnx-light-cpu` links the *same* `lib_onnx_core` that the running
> `onnx_light` package uses. Building the integration from an onnx-light source
> tree (`--onnx-light-source`) while a *separately installed* onnx-light Python
> package runs the model creates two independent dispatch tables, so the
> registration populates one the evaluator never reads. Build with
> `--onnx-light` (`find_package`) so both share the installed shared library.
> See the "Registering kernels" documentation page for details.

For a native C++ integration, build with `-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`
(requires the [onnx-light](https://github.com/xadupre/onnx-light) C++ package).
This builds `lib_onnx_light_cpu_kernels`, whose adapters derive from
onnx-light's `KernelBase`. Calling `onnx_light_cpu::RegisterAllKernels()`
installs the complete inventory into onnx-light's shared kernel dispatch table;
families can also be registered directly, for example with
`RegisterNormalizationKernels()`:

```cpp
#include <onnx_light_cpu/kernels/register_kernels.h>

onnx_light_cpu::RegisterKernelGlobal("", "Abs");
onnx_light_cpu::RegisterAllKernelsGlobal();
onnx_light_cpu::RegisterKernelForSession(runtime_context, "", "Abs");
onnx_light_cpu::RegisterAllKernelsForSession(runtime_context);
```

See the
[standalone C++ inference example](examples/cpp/standalone_inference/README.md)
for a separate CMake project that loads or creates an ONNX model, executes it
through `onnx-light`, and confirms that an `onnx-light-cpu` kernel was used.

The same registration is exposed to Python (in builds compiled with the
onnx-light integration) as `onnx_light_cpu.onnx_py._cpuregister.register_all_kernels()`.

Performance-oriented kernels use the available SIMD paths and, where
applicable, onnx-light's shared CPU executor. Portable kernels retain scalar
fallbacks for every supported platform and data type.

## Testing

### C++ tests

```bash
cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_TESTS=ON \
      -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

### Python tests

```bash
pip install -e .[dev]
pytest unittests/python/
```

## Documentation

The full documentation is published at
[xadupre.github.io/docs/onnx-light-cpu](https://xadupre.github.io/docs/onnx-light-cpu/).

It is built with [Sphinx](https://www.sphinx-doc.org/) and
includes an auto-generated table of the available kernels plus a runnable
example gallery:

```bash
pip install -e .[docs]
sphinx-build -b html docs dist/html
```

## Architecture

The kernel uses runtime CPU feature detection (CPUID on x86) to select the
optimal SIMD implementation:

1. **AVX-512F** (512-bit): Processes 16 float32s / 8 float64s per iteration
2. **AVX2** (256-bit): Processes 8 float32s / 4 float64s per iteration, native `pabsd` for int32
3. **AVX** (256-bit): Processes 8 float32s / 4 float64s per iteration
4. **SSE2** (128-bit): Processes 4 float32s / 2 float64s per iteration
5. **Scalar**: Standard C++ fallback for non-x86 platforms

The detection result is cached in a static variable (thread-safe due to C++11
static initialization guarantees), so the dispatch overhead is paid only once.

## License

Apache-2.0. See [LICENSE](https://github.com/xadupre/onnx-light-cpu/blob/main/LICENSE).

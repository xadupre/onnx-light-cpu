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
- Python ≥ 3.10
- [nanobind](https://github.com/wjakob/nanobind) ≥ 1.3.2

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

### Pure CMake (C++ only)

```bash
cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_TESTS=ON \
      -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build
```

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

`register_kernels` installs the SIMD-accelerated kernels into
[onnx-light](https://github.com/xadupre/onnx-light)'s shared C++ kernel
dispatch table so any ONNX model using `Abs`, `Exp`, `Log`, `Gemm` or `Not`
runs the optimized kernel when evaluated through a `ReferenceEvaluator`:

```python
import numpy as np
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import register_kernels

register_kernels()  # installs the kernels into onnx-light's dispatch table
sess = ReferenceEvaluator(model)  # any model containing an Abs node
(y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.0], dtype=np.float32)})
```

`register_kernels` is only available in builds compiled with the onnx-light
integration (`-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`); it wraps the compiled
`onnx_light_cpu.onnx_py._cpuregister.register_all_kernels()` binding.

For a native C++ integration, build with `-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`
(requires the [onnx-light](https://github.com/xadupre/onnx-light) C++ package).
This builds `lib_onnx_light_cpu_kernels`, which exposes `onnx_light_cpu::AbsKernel`,
`onnx_light_cpu::ExpKernel`, `onnx_light_cpu::LogKernel` and
`onnx_light_cpu::NotKernel` classes deriving from
onnx-light's `KernelBase`. Calling `onnx_light_cpu::RegisterAllKernels()`
installs all of them into onnx-light's shared kernel dispatch table (or call the
per-operator `RegisterAbsKernel()`, `RegisterExpKernel()`, `RegisterLogKernel()`
and `RegisterNotKernel()` functions individually) so every
`Abs`/`Exp`/`Log`/`Not` node runs the SIMD kernel:

```cpp
#include <onnx_light_cpu/kernels/register_kernels.h>

onnx_light_cpu::RegisterAllKernels();  // Abs/Exp/Log/Not now use the SIMD kernels
```

The same registration is exposed to Python (in builds compiled with the
onnx-light integration) as `onnx_light_cpu.onnx_py._cpuregister.register_all_kernels()`.

When running through onnx-light, these kernels combine SIMD with
multithreading: each kernel splits its work across onnx-light's shared
`ParallelFor` thread pool. The pool sizes itself to the number of hardware
threads, and its grain-size threshold keeps small tensors on a single thread
(SIMD only), so large arrays are parallelized while small ones avoid
thread-dispatch overhead.

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

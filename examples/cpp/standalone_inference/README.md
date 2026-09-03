# Standalone C++ inference

This independent CMake project runs an ONNX model with the kernels registered
by `onnx-light-cpu`. With no argument, it creates and runs a small `Abs` model.
It can also load a compatible ONNX file with one FLOAT input of shape `[4]`
and one FLOAT output:

```bash
cmake -S examples/cpp/standalone_inference -B build-standalone \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/path/to/install/prefix
cmake --build build-standalone
./build-standalone/onnx_light_cpu_inference
./build-standalone/onnx_light_cpu_inference model.onnx
```

The prefix must contain C++ installations of both `onnx-light` and
`onnx-light-cpu`, with the latter built using
`-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`. The executable verifies both the
numerical output and that `onnx_light_cpu::Abs` handled the model node.

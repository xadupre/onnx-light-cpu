"""
onnx-light-cpu provides highly optimized CPU kernels for onnx-light.

Supported operations: Abs (float16, float32, float64, int8, int32, int64),
Exp and Log (float16, float32, float64), and Not (bool).
Uses AVX/AVX2/AVX-512 when available, with SSE2 and scalar fallbacks.

The :func:`register_kernels` helper installs these kernels into ``onnx-light``'s
C++ dispatch table so any ONNX model using ``Abs``, ``Exp``, ``Log``, ``Gemm``
or ``Not`` runs the optimized kernel when evaluated through a
``ReferenceEvaluator``.
"""

from ._processor_profile import (
    BandwidthMeasurement,
    CacheDescriptor,
    ComputeMeasurement,
    ExplicitAffinity,
    LatencyMeasurement,
    MemoryLevelMeasurement,
    ProcessorPerformanceProfile,
    ProcessorProfileMetadata,
    ProcessorProfileOptionsEcho,
    ProcessorProfileTopology,
    RooflineMeasurement,
    benchmark_processor_performance,
)
from ._register import (
    RegisteredKernel,
    clear_used_kernel_names,
    has_backend_test_cases,
    register_backend_test_cases,
    register_kernels,
    registered_kernel_names,
    registered_kernels,
    used_kernel_names,
)

__version__ = "0.1.16"

__all__ = [
    "BandwidthMeasurement",
    "CacheDescriptor",
    "ComputeMeasurement",
    "ExplicitAffinity",
    "LatencyMeasurement",
    "MemoryLevelMeasurement",
    "ProcessorPerformanceProfile",
    "ProcessorProfileMetadata",
    "ProcessorProfileOptionsEcho",
    "ProcessorProfileTopology",
    "RegisteredKernel",
    "RooflineMeasurement",
    "benchmark_processor_performance",
    "clear_used_kernel_names",
    "has_backend_test_cases",
    "register_backend_test_cases",
    "register_kernels",
    "registered_kernel_names",
    "registered_kernels",
    "used_kernel_names",
]

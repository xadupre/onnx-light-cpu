"""
onnx-light-cpu provides highly optimized CPU kernels for onnx-light.

The available operators and data types are exposed by
:func:`registered_kernels`. Kernels use AVX/AVX2/AVX-512 when available, with
SSE2 and scalar fallbacks.

The :func:`register_kernels` helper installs these kernels into ``onnx-light``'s
C++ dispatch table so supported ONNX nodes run the optimized kernels when
evaluated through a ``ReferenceEvaluator``.
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
    OperatorSupport,
    RegisteredKernel,
    clear_used_kernel_names,
    custom_op_schemas,
    has_backend_test_cases,
    operator_support,
    operator_schema_lookup,
    register_backend_test_cases,
    register_custom_gradients,
    register_kernels,
    register_operator_support,
    registered_kernel_names,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)
from ._simd import SimdLevel, detect_simd_level, has_cpu_kernels

__version__ = "0.1.16"

__all__ = [
    "BandwidthMeasurement",
    "CacheDescriptor",
    "ComputeMeasurement",
    "ExplicitAffinity",
    "LatencyMeasurement",
    "MemoryLevelMeasurement",
    "OperatorSupport",
    "ProcessorPerformanceProfile",
    "ProcessorProfileMetadata",
    "ProcessorProfileOptionsEcho",
    "ProcessorProfileTopology",
    "RegisteredKernel",
    "RooflineMeasurement",
    "SimdLevel",
    "benchmark_processor_performance",
    "clear_used_kernel_names",
    "custom_op_schemas",
    "detect_simd_level",
    "has_backend_test_cases",
    "has_cpu_kernels",
    "operator_schema_lookup",
    "operator_support",
    "register_backend_test_cases",
    "register_custom_gradients",
    "register_kernels",
    "register_operator_support",
    "registered_kernel_names",
    "registered_kernels",
    "set_kernel_usage_recording",
    "used_kernel_names",
]

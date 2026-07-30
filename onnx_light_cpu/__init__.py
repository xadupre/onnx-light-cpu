"""
onnx-light-cpu provides highly optimized CPU kernels for onnx-light.

Supported operations: Abs (float32, float64, int32, int64).
Uses AVX/AVX2/AVX-512 when available, with SSE2 and scalar fallbacks.

The :func:`register_kernels` helper plugs these kernels into an ``onnx-light``
``ReferenceEvaluator`` so any ONNX model using ``Abs`` runs the optimized
kernel.
"""

from ._register import register_kernels

__version__ = "0.1.0"

__all__ = ["register_kernels"]

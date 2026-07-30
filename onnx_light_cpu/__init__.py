"""
onnx-light-cpu provides highly optimized CPU kernels for onnx-light.

Supported operations: Abs (float32, float64, int32, int64).
Uses AVX/AVX2/AVX-512 when available, with SSE2 and scalar fallbacks.
"""

__version__ = "0.1.0"

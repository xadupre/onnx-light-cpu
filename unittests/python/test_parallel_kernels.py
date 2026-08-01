# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Concurrency tests for the SIMD kernels.

The Python bindings release the GIL while a kernel runs so that several threads
can execute a kernel on disjoint chunks of an array at the same time. These
tests exercise that path: they run the kernels from a thread pool and check the
results still match the single-threaded reference, guarding against data races
introduced by the GIL release.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from onnx_light_cpu.onnx_py._cpukernels import (
    abs as cpu_abs,
    exp as cpu_exp,
    log as cpu_log,
    logical_not,
)


def _run_concurrently(kernel, arrays, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(kernel, arrays))


class TestParallelKernels:
    def test_abs_chunks_match_single_thread(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-100.0, 100.0, size=1_000_003).astype(np.float32)
        chunks = np.array_split(x, 4)
        results = _run_concurrently(cpu_abs, chunks)
        np.testing.assert_array_equal(np.concatenate(results), np.abs(x))

    def test_exp_chunks_match_single_thread(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-5.0, 5.0, size=500_003).astype(np.float32)
        chunks = np.array_split(x, 4)
        results = _run_concurrently(cpu_exp, chunks)
        np.testing.assert_allclose(np.concatenate(results), np.exp(x), rtol=1e-4, atol=1e-5)

    def test_log_chunks_match_single_thread(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0.1, 100.0, size=500_003).astype(np.float32)
        chunks = np.array_split(x, 4)
        results = _run_concurrently(cpu_log, chunks)
        np.testing.assert_allclose(np.concatenate(results), np.log(x), rtol=1e-4, atol=1e-5)

    def test_logical_not_chunks_match_single_thread(self):
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=1_000_003).astype(np.bool_)
        chunks = np.array_split(x, 4)
        results = _run_concurrently(logical_not, chunks)
        np.testing.assert_array_equal(np.concatenate(results), np.logical_not(x))

    def test_many_threads_same_input(self):
        # Running the same input from many threads must not corrupt any result.
        rng = np.random.default_rng(4)
        x = rng.uniform(-100.0, 100.0, size=200_003).astype(np.float32)
        expected = np.abs(x)
        results = _run_concurrently(cpu_abs, [x] * 16, max_workers=8)
        for out in results:
            np.testing.assert_array_equal(out, expected)

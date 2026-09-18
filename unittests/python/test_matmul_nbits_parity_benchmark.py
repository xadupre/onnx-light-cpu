# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Packed-constant INT4 parity; enable the full model matrix with OLC_MUSE_INT4_FULL=1."""

import os
import unittest

from tools.benchmark_matmul_nbits_parity import (
    BLOCK_SIZE,
    DTYPES,
    ROWS,
    VOCABULARY,
    fingerprint,
    make_inputs,
    make_model,
    memory_accounting,
    parse_args,
    projections,
    run_case,
)


class TestMatMulNBitsBenchmarkContract(unittest.TestCase):
    def test_exact_projection_families(self):
        self.assertEqual(
            projections(),
            {
                "q": (6656, 4096),
                "k": (6656, 256),
                "v": (6656, 256),
                "attention_output": (4096, 6656),
                "gate": (6656, 19968),
                "up": (6656, 19968),
                "down": (19968, 6656),
                "lm_head": (6656, VOCABULARY),
            },
        )
        self.assertEqual(VOCABULARY, 202048)
        self.assertEqual(ROWS, (1, 8, 128))
        self.assertEqual(DTYPES, ("float32", "float16", "bfloat16"))

    def test_workspace_bound_is_independent_of_projection_and_dtype(self):
        for dtype in DTYPES:
            for m in ROWS:
                for k, n in projections().values():
                    with self.subTest(dtype=dtype, m=m, k=k, n=n):
                        memory = memory_accounting(m, k, n, dtype)
                        self.assertEqual(memory["kernel_panel_workspace_bytes_per_worker"], 6144)
                        self.assertEqual(memory["kernel_full_weight_copy_bytes_per_run"], 0)
                        self.assertEqual(
                            memory["kernel_full_input_conversion_copy_bytes_per_run"], 0
                        )
                        self.assertEqual(memory["packed_initializer_bytes"], k * n // 2)
                        self.assertEqual(memory["avoided_full_float_weight_bytes"], k * n * 4)
                        self.assertIn("analytical", memory["kind"])
                        self.assertNotIn("rss", memory)

    def test_cli_defaults_are_small_and_expensive_matrix_is_explicit(self):
        args = parse_args([])
        self.assertFalse(args.full)
        self.assertEqual(args.threads, 1)
        self.assertGreater(args.repeat, 1)
        args = parse_args(["--full", "--projection", "down", "--m", "128", "--dtype", "float16"])
        self.assertTrue(args.full)
        self.assertEqual(args.projection, ["down"])
        self.assertEqual(args.m, [128])
        self.assertEqual(args.dtype, ["float16"])


class TestMatMulNBitsConstantParity(unittest.TestCase):
    def test_deterministic_packed_initializers_not_graph_inputs(self):
        import numpy as np
        from onnx_light.onnx import TensorProto

        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                a, packed, scales = make_inputs(8, 65, 33, dtype)
                self.assertEqual(
                    fingerprint(a, packed, scales), fingerprint(*make_inputs(8, 65, 33, dtype))
                )
                self.assertEqual(packed.dtype, np.uint8)
                self.assertEqual(packed.shape, (33, 3, BLOCK_SIZE // 2))
                self.assertEqual(scales.shape, (33, 3))
                self.assertLess(packed.nbytes + scales.nbytes, 65 * 33 * 4)
                model = make_model(8, 65, 33, dtype, packed, scales)
                self.assertEqual([value.name for value in model.graph.input], ["A"])
                self.assertEqual(
                    [value.name for value in model.graph.initializer], ["B", "scales"]
                )
                self.assertEqual(model.graph.initializer[0].data_type, TensorProto.UINT8)
                self.assertEqual(list(model.graph.initializer[0].dims), list(packed.shape))
                self.assertEqual(len(model.graph.node), 1)
                self.assertEqual(model.graph.node[0].op_type, "MatMulNBits")

    def test_ort_parity_constant_reuse_and_panel_tails(self):
        for dtype in DTYPES:
            for m, k, n in ((1, 33, 31), (8, 65, 32), (9, 97, 33), (128, 64, 17)):
                with self.subTest(dtype=dtype, m=m, k=k, n=n):
                    result = run_case("panel_tail", m, k, n, dtype, repeat=2, warmup=0)
                    self.assertTrue(result["constants_unchanged"])
                    self.assertGreater(len(result["cpu_samples_seconds"]), 0)
                    self.assertGreater(result["cpu_session_preparation_seconds"], 0)
                    if dtype == "bfloat16" and result["ort_bfloat16_rejection"]:
                        self.assertIn("BF16-rounded", result["oracle"])

    def test_muse_key_projection_all_rows_and_dtypes(self):
        for dtype in DTYPES:
            for m in ROWS:
                with self.subTest(dtype=dtype, m=m):
                    result = run_case("k", m, *projections()["k"], dtype, repeat=2, warmup=0)
                    self.assertTrue(result["constants_unchanged"])

    @unittest.skipUnless(os.environ.get("OLC_MUSE_INT4_FULL") == "1", "expensive model matrix")
    def test_full_muse_projection_matrix(self):
        for name, (k, n) in projections().items():
            for m in ROWS:
                for dtype in DTYPES:
                    with self.subTest(projection=name, dtype=dtype, m=m):
                        result = run_case(name, m, k, n, dtype, repeat=2, warmup=0)
                        self.assertTrue(result["constants_unchanged"])


if __name__ == "__main__":
    unittest.main()

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session-owned kernel usage recording with real native evaluators."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernel_for_session,
    set_kernel_usage_recording,
    used_kernel_names,
)


class TestKernelUsage(ExtTestCase):
    @staticmethod
    def _session(op_type="Abs"):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node(op_type, ["X"], ["Y"])],
                "kernel_usage",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        session = ReferenceEvaluator(model, cpu_execution={"num_threads": 1})
        register_kernel_for_session(session, "", op_type)
        return session

    @staticmethod
    def _run(session):
        return session.run(None, {"X": np.array([-1, 2], dtype=np.float32)})[0]

    def test_recording_lifecycle_and_snapshot(self):
        session = self._session()
        np.testing.assert_array_equal(self._run(session), [1, 2])
        self.assertEqual(used_kernel_names(session), [])

        set_kernel_usage_recording(session, True)
        self._run(session)
        snapshot = used_kernel_names(session)
        self.assertEqual(snapshot, ["onnx_light_cpu::Abs"])
        snapshot.append("not a recorded kernel")
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

        clear_used_kernel_names(session)
        self.assertEqual(used_kernel_names(session), [])
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

        set_kernel_usage_recording(session, False)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])
        clear_used_kernel_names(session)
        self._run(session)
        self.assertEqual(used_kernel_names(session), [])
        set_kernel_usage_recording(session, True)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

    def test_independent_sessions_do_not_share_recording_state(self):
        first = self._session()
        set_kernel_usage_recording(first, True)
        self._run(first)
        second = self._session("Exp")
        self._run(second)
        self.assertEqual(used_kernel_names(first), ["onnx_light_cpu::Abs"])
        self.assertEqual(used_kernel_names(second), [])

        set_kernel_usage_recording(second, True)
        self._run(second)
        clear_used_kernel_names(first)
        set_kernel_usage_recording(first, False)
        self._run(first)
        self._run(second)
        self.assertEqual(used_kernel_names(first), [])
        self.assertEqual(used_kernel_names(second), ["onnx_light_cpu::Exp"] * 2)

        clear_used_kernel_names(second)
        set_kernel_usage_recording(first, True)
        self._run(first)
        self.assertEqual(used_kernel_names(first), ["onnx_light_cpu::Abs"])
        self.assertEqual(used_kernel_names(second), [])

    def test_concurrent_sessions_clear_only_their_own_log(self):
        first = self._session()
        second = self._session("Exp")
        set_kernel_usage_recording(first, True)
        set_kernel_usage_recording(second, True)
        barrier = Barrier(2, timeout=30)

        def run(session, clear):
            for index in range(32):
                barrier.wait()
                self._run(session)
                if clear and index == 15:
                    clear_used_kernel_names(session)
            return used_kernel_names(session)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_result = executor.submit(run, first, True)
            second_result = executor.submit(run, second, False)
            self.assertEqual(first_result.result(), ["onnx_light_cpu::Abs"] * 16)
            self.assertEqual(second_result.result(), ["onnx_light_cpu::Exp"] * 32)

    def test_recording_retains_first_1024_invocations_until_clear(self):
        session = self._session()
        set_kernel_usage_recording(session, True)
        for _ in range(1030):
            self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"] * 1024)
        clear_used_kernel_names(session)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

    def test_recording_requires_an_explicit_valid_session(self):
        for function in (used_kernel_names, clear_used_kernel_names):
            with self.subTest(function=function.__name__), self.assertRaises(TypeError):
                function()
        with self.assertRaises(TypeError):
            set_kernel_usage_recording()
        with self.assertRaises(TypeError):
            set_kernel_usage_recording(True)
        for session in (None, object(), True, SimpleNamespace(_ctx=object(), _runner=object())):
            for function, args in (
                (used_kernel_names, (session,)),
                (clear_used_kernel_names, (session,)),
                (set_kernel_usage_recording, (session, True)),
            ):
                with (
                    self.subTest(function=function.__name__, session=session),
                    self.assertRaises(TypeError),
                ):
                    function(*args)

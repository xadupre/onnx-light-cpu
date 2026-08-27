---
name: add-custom-operator
description: Add an operator from a custom ONNX domain to onnx-light-cpu end to end.
---

# Add a custom-domain operator

Use this skill when adding an operator outside the standard ONNX domains. Treat the
operator's upstream implementation as the compatibility contract and complete every
applicable surface below in one change.

1. Record the exact domain, capitalization, first opset, attributes, defaults, input
   broadcasting, supported types, numerical edge cases, and upstream reference.
2. Add a `LightOpSchema` provider. Include attributes and type constraints, expose it
   under `schemas/<normalized_domain>/`, expose it through Python, and compose it with
   the standard schema lookup used by `GraphBuilder`.
3. Add a symbolic shape function. Validate ranks, types, and known incompatible
   dimensions; preserve symbolic dimensions and add equality constraints where needed.
   Keep it under `shapes/<normalized_domain>/`.
4. Add a peak-memory function for every supported device. Return scratch memory only,
   excluding inputs and outputs; use zero explicitly for allocation-free kernels.
5. Implement the portable scalar kernel first. Use `ExecuteRanges`, checked size
   arithmetic, repository half conversions, precise errors, and no ISA-specific code
   until scalar parity and benchmarks exist. Put low-level code in
   `impl/<normalized_domain>/` and runtime kernels in `kernels/<normalized_domain>/`.
6. Add tuning keys, validated defaults, calibration candidates when useful, and an ABI
   bump whenever parameter meaning changes.
7. Register the kernel with its custom domain, exact opset bounds, device, and complete
   type list. Update structured registration inventories and generated kernel docs.
8. Add backend TEST and BENCHMARK cases for every supported type, attribute mode,
   boundary shape, empty-input contract, and invalid shape that the harness can express.
   Keep domain-specific cases in `backend_test/<normalized_domain>/`.
9. Add low-level, runtime-dispatch, tuning, shape, memory, and schema tests. Compare
   numerical results with the upstream runtime and include symbolic-shape coverage.
10. Add reverse-mode gradient rules without mutating the default registry. Cover every
    differentiable input, broadcasting reductions, zero/singular cases, and both forward
    attribute modes. Keep them in `gradient/<normalized_domain>/`.
11. Add only safe graph fusions: require exact topology and attributes, exclusive
    intermediates, compatible types/shapes, and the custom-domain opset import. Keep the
    unfused graph as the fallback, test rejection guards, and store the patterns in
    `patterns/<normalized_domain>/`.
12. Add one runnable gallery example and one bounded benchmark against the upstream
    runtime. Verify output parity before timing and reduce sizes under `UNITTEST_GOING`.
13. Update public API documentation and package descriptions. Run clang-format, Ruff,
    Black, targeted C++/Python tests, then the registration and gallery end-to-end tests.

Never advertise schema types, metrics, gradients, or fusion variants that the kernel and
tests do not implement.

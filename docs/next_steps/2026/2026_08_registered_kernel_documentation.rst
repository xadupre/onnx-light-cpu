Registered kernel documentation
===============================

:Date: 2026-08

**complete**

Issue `#258 <https://github.com/xadupre/onnx-light-cpu/issues/258>`_ replaces
the manually maintained and source-scanned kernel inventories with metadata
collected from the registrations that the runtime actually executes. The same
inventory will be exposed to Python and will generate one stable documentation
page per registered kernel.

Target architecture
-------------------

The C++ registration path is the single source of truth. Every
``Register*Kernel[s]`` function calls a common helper carrying the ONNX domain,
operator, device, C++ kernel name, supported element types, and optional opset
bounds. In normal mode the helper updates onnx-light's dispatch table. In
inventory mode it records that metadata without mutating runtime state.

``RegisterAllKernels()`` is executed in inventory mode to produce a sorted list.
The Python binding exposes that list as immutable structured records through
``onnx_light_cpu.registered_kernels()``. The existing
``registered_kernel_names()`` compatibility API is derived from those records
instead of maintaining another operator list.

Sphinx consumes only the public Python API. It creates an index and one RST
page per registration before source reading, uses deterministic filenames and
ordering, and removes stale generated pages. The current C++ source scanner is
deleted.

Implementation sequence
------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 35 18 12

   * - Step
     - Work
     - Completion gate
     - Depends on
     - Status
   * - 1
     - Add the common C++ registration helper and structured metadata record.
       Route every existing kernel registration through it.
     - Runtime registration behavior is unchanged. Inventory collection returns
       every registration in deterministic order and rejects duplicate
       ``(domain, operator, device)`` entries.
     - None
     - Complete
   * - 2
     - Expose ``registered_kernels()`` in the compiled binding and public Python
       package. Derive ``registered_kernel_names()`` from it.
     - Python returns domain, operator, device, kernel name, types, and optional
       version bounds for every C++ registration.
     - Step 1
     - Complete
   * - 3
     - Replace the Sphinx source scanner with API-driven page and index
       generation, including stale-page removal.
     - Two consecutive generations are byte-identical. Adding, renaming, or
       removing a registration changes the generated pages without editing a
       documentation-side operator list.
     - Step 2
     - Complete
   * - 4
     - Add C++, Python, API-documentation, and Sphinx parity tests.
     - A clean warnings-as-errors Sphinx build contains exactly one page per
       inventory record; focused C++ and Python suites pass.
     - Steps 1 through 3
     - Complete

Step 1 (this PR) adds ``onnx_light_cpu::KernelRegistration`` (the structured
domain/operator/device/kernel-name/types/opset-bounds record) together with
``RegisterKernel`` and ``CollectRegisteredKernels`` in
``onnx_light_cpu/kernels/kernel_registration.{h,cc}``. Every
``Register*Kernel[s]`` function now builds one such record and hands it, with
its node factory, to ``RegisterKernel`` instead of calling onnx-light's
``RegisterKernelFn`` directly. ``CollectRegisteredKernels`` runs
``RegisterAllKernels`` with an inventory scope active, so it never mutates
onnx-light's shared ``KernelDispatchTable``, and returns the metadata sorted
by ``(domain, operator, device, kernel_name)``. A registration pass rejects a
repeated ``(domain, operator, device)`` key with ``std::invalid_argument``.

Step 2 (this PR) exposes that inventory to Python. The compiled
``_cpuregister`` extension gains a ``registered_kernels()`` binding that calls
``CollectRegisteredKernels()`` and renders each record's ``Device`` and
``TensorProto::DataType`` fields as strings. The public
``onnx_light_cpu`` package wraps each tuple into an immutable
``RegisteredKernel`` (``typing.NamedTuple``) record through
``onnx_light_cpu.registered_kernels()``, and
``onnx_light_cpu.registered_kernel_names()`` (and the C++
``RegisteredKernelNames()`` it is built on) is now derived from those same
records instead of a separately hand-maintained ``op_type -> kernel name``
list.

Step 3 (this PR) replaces the C++ source scanner with generation driven
exclusively by ``onnx_light_cpu.registered_kernels()``. The
``docs/_ext/kernel_scan.py`` and ``docs/_ext/onnx_kernels.py`` modules (and
the ``registered-kernels`` directive they provided) are deleted. The new
``docs/_ext/kernel_pages.py`` extension connects to Sphinx's
``builder-inited`` event and, before any source file is read, writes one RST
page per registration plus a deterministic index under
``docs/kernels_generated/`` (an entirely build-generated, ``.gitignore``-d
directory that ``docs/kernels.rst`` links to through a toctree). Filenames
are derived from ``(domain, op_type, device)`` and disambiguated in the same
stable order on the rare slugify collision, so two consecutive generations
are byte-identical; any page left over from a renamed or removed registration
is deleted so the directory always matches the current inventory exactly.

Step 4 (this PR) adds the final parity coverage across the whole pipeline,
on top of the C++ (``unittests/cc/test_onnx_light_kernel_registration.cc``,
``unittests/cc/test_onnx_light_kernel_usage.cc``) and Python
(``unittests/python/test_kernels_doc.py``, ``unittests/python/test_kernels_e2e.py``,
``unittests/python/test_api_doc.py``) suites steps 1 through 3 already added.
This PR extends ``test_onnx_light_kernel_registration.cc`` with a test proving
that optional opset bounds (``since_version``/``until_version``) round-trip
exactly through ``RegisterKernel``/``CollectRegisteredKernels`` alongside the
existing deterministic-ordering, no-mutation, and duplicate-rejection
coverage. It also extends ``test_kernels_doc.py`` with a new
``TestGenerationParityWithLiveInventory`` class proving the *live*
``onnx_light_cpu.registered_kernels()`` inventory -- not just the fabricated
records the rest of that file already exercised -- produces exactly one
generated page and index entry per record, with content that matches the
record it documents, and regenerates byte-identically. Finally, the ``docs``
workflow now builds Sphinx with ``-W`` (warnings treated as errors), so a
broken cross-reference or any other build warning fails CI instead of
silently degrading the generated documentation; a local ``sphinx-build -W``
run confirmed the ``kernels_generated/`` directory contains exactly one page
per ``registered_kernels()`` record plus the index, with zero warnings.

Validation and compatibility
----------------------------

The implementation must preserve the public ``registered_kernel_names()`` API
while documenting ``registered_kernels()`` as the authoritative inventory.
Collection must not install, replace, or execute kernels. Generated files are
build artifacts rather than hand-edited sources.

Validation covers:

* deterministic C++ ordering and complete metadata;
* parity between C++ registrations and Python records;
* parity between Python records, generated filenames, and index entries;
* stale-page cleanup and filename-collision handling;
* the existing runtime registration and kernel-usage tests;
* a clean Sphinx build with warnings treated as errors.

Known integration risk
----------------------

The compiled registration extension must link the same onnx-light shared
runtime used by Python. Mixing current headers with stale onnx-light libraries
causes unresolved executor symbols, while embedding a separate static runtime
would create a second dispatch table. CI and local validation must therefore
rebuild onnx-light consistently and use the repository's supported integration
configuration.

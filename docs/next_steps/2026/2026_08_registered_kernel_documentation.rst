Registered kernel documentation
===============================

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
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 27 39 20

   * - Step
     - Work
     - Completion gate
     - Depends on
   * - 1
     - Add the common C++ registration helper and structured metadata record.
       Route every existing kernel registration through it.
     - Runtime registration behavior is unchanged. Inventory collection returns
       every registration in deterministic order and rejects duplicate
       ``(domain, operator, device)`` entries.
     - None
   * - 2
     - Expose ``registered_kernels()`` in the compiled binding and public Python
       package. Derive ``registered_kernel_names()`` from it.
     - Python returns domain, operator, device, kernel name, types, and optional
       version bounds for every C++ registration.
     - Step 1
   * - 3
     - Replace the Sphinx source scanner with API-driven page and index
       generation, including stale-page removal.
     - Two consecutive generations are byte-identical. Adding, renaming, or
       removing a registration changes the generated pages without editing a
       documentation-side operator list.
     - Step 2
   * - 4
     - Add C++, Python, API-documentation, and Sphinx parity tests.
     - A clean warnings-as-errors Sphinx build contains exactly one page per
       inventory record; focused C++ and Python suites pass.
     - Steps 1 through 3

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


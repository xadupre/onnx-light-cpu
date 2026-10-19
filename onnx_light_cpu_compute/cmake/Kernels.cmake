file(GLOB_RECURSE CPU_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
# ISA-specific translation units are grouped by directory or filename family.
# New implementations following these conventions automatically inherit the
# corresponding compiler feature checks and flags.
foreach(_source IN LISTS CPU_KERNEL_SOURCES)
  if(_source MATCHES "_ssse3[.]cc$")
    list(APPEND _ssse3_sources "${_source}")
  endif()
  if(_source MATCHES "_avx[.]cc$")
    list(APPEND _avx_sources "${_source}")
  endif()
  if(_source MATCHES "_avx2[.]cc$" AND NOT _source MATCHES "integer_gemm_avx2[.]cc$")
    list(APPEND _avx2_sources "${_source}")
  endif()
  if(_source MATCHES "/avx512/")
    list(APPEND _avx512f_sources "${_source}")
  endif()
  if(_source MATCHES "/avx512dq/")
    list(APPEND _avx512dq_sources "${_source}")
  endif()
  if(_source MATCHES "/logical/avx512/")
    list(APPEND _avx512bw_sources "${_source}")
  endif()
  if(_source MATCHES "/avx512bw/")
    list(APPEND _avx512bw_sources "${_source}")
  endif()
  if(_source MATCHES "/avx512fp16/")
    list(APPEND _avx512fp16_sources "${_source}")
  endif()
  if(_source MATCHES "/avx512bf16/")
    list(APPEND _avx512bf16_sources "${_source}")
  endif()
  if(_source MATCHES "avx512vnni")
    list(APPEND _avx512vnni_sources "${_source}")
    if(_source MATCHES "/avx512bw/")
      list(APPEND _avx512bw_vnni_sources "${_source}")
    endif()
    if(_source MATCHES "/math/gemm/")
      list(APPEND _gemm_avx512vnni_sources "${_source}")
    endif()
  endif()
  if(_source MATCHES "avx2_fma[.]cc$")
    list(APPEND _avx2_fma_sources "${_source}")
  endif()
  if(_source MATCHES "/math/gemm/avx2/")
    list(APPEND _gemm_avx2_fma_sources "${_source}")
  endif()
  if(_source MATCHES "integer_gemm_avx2[.]cc$")
    list(APPEND _avx2_integer_sources "${_source}")
  endif()
  if(_source MATCHES "(integer_gemm|matmul_nbits_int8)_avxvnni[.]cc$")
    list(APPEND _avxvnni_integer_sources "${_source}")
  endif()
  if(_source MATCHES "f16c[.]cc$")
    list(APPEND _avx_f16c_sources "${_source}")
  endif()
  if(_source MATCHES "avx2_f16c[.]cc$")
    list(APPEND _avx2_f16c_sources "${_source}")
  endif()
  if(_source MATCHES "/amx/.*tile[.]cc$")
    list(APPEND _amx_tile_sources "${_source}")
  endif()
  if(_source MATCHES "/amx/.*bf16[.]cc$")
    list(APPEND _amx_bf16_sources "${_source}")
  endif()
  if(_source MATCHES "/amx/.*int8[.]cc$")
    list(APPEND _amx_int8_sources "${_source}")
  endif()
  if(_source MATCHES "/arm/.*neon[.]cc$")
    list(APPEND _neon_sources "${_source}")
  endif()
  if(_source MATCHES "/arm/.*sve[.]cc$")
    list(APPEND _sve_sources "${_source}")
  endif()
  if(_source MATCHES "/arm/.*neon_dotprod[.]cc$")
    list(APPEND _neon_dotprod_sources "${_source}")
  endif()
endforeach()

if(NOT ONNX_LIGHT_CPU_HAVE_SSSE3_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_ssse3_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX2_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx2_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  # Exclude AVX-512F translation units rather than fail on unsupported targets.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512f_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512dq_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES
      ${_avx2_fma_sources}
      ${_gemm_avx2_fma_sources}
      ${_avx2_integer_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx_f16c_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX2_FLAG OR NOT ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx2_f16c_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512bw_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG)
  # Exclude the native AVX-512FP16 GEMM kernel when the compiler cannot target
  # the ISA; the FLOAT16 path keeps its converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512fp16_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG)
  # Exclude the native AVX-512BF16 GEMM kernel when the compiler cannot target
  # the ISA; the BFLOAT16 path keeps its converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512bf16_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
  # Exclude the native AVX-512 VNNI INT8 dot-product when the compiler cannot
  # target the ISA; the integer matmul keeps its AVX2/scalar fallbacks.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avx512vnni_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVXVNNI_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_avxvnni_integer_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG)
  # Exclude the native AMX-BF16 GEMM kernel when the compiler cannot target the
  # ISA; the BFLOAT16 path keeps its AVX-512BF16 / converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_amx_bf16_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
  # Exclude the native AMX-INT8 GEMM kernel when the compiler cannot target the
  # ISA; the integer matmul keeps its AVX-512 VNNI / AVX2 / scalar fallbacks.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_amx_int8_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_neon_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_sve_sources})
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES ${_neon_dotprod_sources})
endif()


add_library(lib_onnx_light_cpu ${_cpu_lib_type} ${CPU_KERNEL_SOURCES})
target_include_directories(lib_onnx_light_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_property(TARGET lib_onnx_light_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

if(ONNX_LIGHT_CPU_HAVE_SSSE3_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_ssse3_sources} PROPERTIES COMPILE_OPTIONS "-mssse3")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_HAVE_SSSE3=1)
endif()
target_compile_features(lib_onnx_light_cpu PUBLIC cxx_std_20)
target_compile_definitions(lib_onnx_light_cpu PRIVATE
    ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=${_onnx_light_cpu_max_simd_level_value})

# Thread topology detection and public affinity helpers use the platform
# threading library, but kernel scheduling is owned by onnx-light.
target_link_libraries(lib_onnx_light_cpu PUBLIC Threads::Threads)

if(ONNX_LIGHT_CPU_HAVE_AVX_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx_sources} PROPERTIES COMPILE_OPTIONS "-mavx")
  else()
    set_source_files_properties(${_avx_sources} PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_HAVE_AVX=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx2_sources} PROPERTIES COMPILE_OPTIONS "-mavx2")
  else()
    set_source_files_properties(${_avx2_sources} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_HAVE_AVX2=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx512f_sources} PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
  else()
    set_source_files_properties(${_avx512f_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
  endif()

  if(ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG)
    if(NOT MSVC)
      set_source_files_properties(${_avx512dq_sources} PROPERTIES
          COMPILE_OPTIONS "-mavx512f;-mavx512dq")
    else()
      set_source_files_properties(${_avx512dq_sources} PROPERTIES
          COMPILE_OPTIONS "/arch:AVX512")
    endif()
    target_compile_definitions(lib_onnx_light_cpu PRIVATE
        ONNX_LIGHT_CPU_HAVE_AVX512DQ=1)
  endif()
  # Tell gemm_kernel.cc (compiled at the baseline ISA)
  # that GemmMicroKernel_AVX512_F32/F64 are available to dispatch to. The
  # register-resident compute-profile FP32/FP64 kernel (Processor Profile
  # PR03) and the binary arithmetic SIMD kernels (Binary PR02) dispatch on the
  # same definition.
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_HAVE_AVX512=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG)
  # The native FLOAT16 GEMM kernel needs -mavx512fp16 for the _Float16 type and
  # the vcvtph2psx conversion; the rest of the library keeps the baseline flags
  # and only dispatches here once CpuSupportsAvx512Fp16() reports the ISA. The
  # register-resident compute-profile FP16 throughput kernel (Processor
  # Profile PR03) needs the same flag for native ``vfmadd*ph`` arithmetic.
  set_source_files_properties(${_avx512fp16_sources}
      PROPERTIES COMPILE_OPTIONS "-mavx512fp16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512FP16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG)
  # The native BFLOAT16 GEMM kernel needs -mavx512bf16 for the __m512bh type and
  # the vdpbf16ps dot-product (plus -mavx512bw for the 512-bit zero-extend used
  # to build the pair layout); the rest of the library keeps the baseline flags
  # and only dispatches here once CpuSupportsAvx512Bf16() reports the ISA. The
  # register-resident compute-profile BFLOAT16 dot-product kernel (Processor
  # Profile PR03) needs the same flags.
  set_source_files_properties(${_avx512bf16_sources}
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512bf16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512BF16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx2_fma_sources}
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  else()
    set_source_files_properties(${_avx2_fma_sources}
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)
  if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG AND NOT MSVC)
    set_property(SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/math/avx2/exp_log_kernel_avx2_fma.cc"
        APPEND PROPERTY COMPILE_OPTIONS "-mf16c")
  endif()

  if(NOT MSVC)
    set_source_files_properties(${_avx2_integer_sources} PROPERTIES
        COMPILE_OPTIONS "-mavx2")
  else()
    set_source_files_properties(${_avx2_integer_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
  # The native INT8 VNNI dot-product needs -mavx512vnni for the vpdpbusd
  # (_mm512_dpbusd_epi32) intrinsic and -mavx512f for the ZMM load / reduce; the
  # rest of the library keeps the baseline flags and only dispatches here once
  # CpuSupportsAvx512Vnni() reports the ISA. The register-resident
  # compute-profile INT8 dot-product kernel (Processor Profile PR03) needs the
  # same flags.
  set_source_files_properties(${_avx512vnni_sources}
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vnni")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512VNNI=1)
endif()
if(ONNX_LIGHT_CPU_HAVE_AVXVNNI_FLAG)
  set_source_files_properties(${_avxvnni_integer_sources} PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mavxvnni")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVXVNNI=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG)
  # The AMX tile-state lifecycle needs -mamx-tile for the LDTILECFG/TILERELEASE
  # tile intrinsics; the rest of the library keeps the baseline flags and the
  # module only touches the tiles once AmxTileStateAvailable() reports the ISA.
  set_source_files_properties(${_amx_tile_sources} PROPERTIES
      COMPILE_OPTIONS "-mamx-tile")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_TILE=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG)
  # The native AMX-BF16 GEMM kernel needs -mamx-tile for the tile load/store and
  # -mamx-bf16 for the tdpbf16ps dot-product; the rest of the library keeps the
  # baseline flags and only dispatches here once CpuSupportsAmxBf16() and
  # AmxTileStateAvailable() report the ISA and OS-enabled tile state.
  set_source_files_properties(${_amx_bf16_sources} PROPERTIES
      COMPILE_OPTIONS "-mamx-tile;-mamx-bf16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_BF16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
  # The native AMX-INT8 GEMM kernel needs -mamx-tile for the tile load/store and
  # -mamx-int8 for the tdpbuud dot-product; dispatch additionally checks the ISA
  # and OS-enabled tile state at runtime.
  set_source_files_properties(${_amx_int8_sources} PROPERTIES
      COMPILE_OPTIONS "-mamx-tile;-mamx-int8;-mavx512f;-mavx512bw;-mavx512vnni")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_INT8=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  if(NOT MSVC)
    # Add -mf16c when supported so the packing loops can use the F16C
    # ``vcvtph2ps`` conversion; the AVX2/FMA micro-kernels only require -mavx2.
    if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
      set_source_files_properties(${_gemm_avx2_fma_sources} PROPERTIES
          COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    else()
      set_source_files_properties(${_gemm_avx2_fma_sources} PROPERTIES
          COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
  else()
    set_source_files_properties(${_gemm_avx2_fma_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
  endif()

  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)
  if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
    target_compile_definitions(lib_onnx_light_cpu PRIVATE
        ONNX_LIGHT_CPU_HAVE_F16C=1)
  endif()
endif()

if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx_f16c_sources} PROPERTIES
        COMPILE_OPTIONS "-mavx;-mf16c")
  else()
    set_source_files_properties(${_avx_f16c_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_RMS_F16C=1 ONNX_LIGHT_CPU_HAVE_AVX_F16C=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FLAG AND ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx2_f16c_sources} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mf16c")
  else()
    set_source_files_properties(${_avx2_f16c_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_F16C=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_NEON=1)
  if(ONNX_LIGHT_CPU_HAVE_NEON_FP16_INTRINSICS)
    target_compile_definitions(lib_onnx_light_cpu PRIVATE
        ONNX_LIGHT_CPU_HAVE_NEON_FP16=1)
  endif()
endif()

if(ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS)
  set_source_files_properties(${_sve_sources} PROPERTIES
      COMPILE_OPTIONS "-march=armv8-a+sve")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_SVE=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
  set_source_files_properties(${_neon_dotprod_sources}
      PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG)
  if(NOT MSVC)
    set_source_files_properties(${_avx512bw_sources} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw")
    if(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
      set_property(SOURCE ${_gemm_avx512vnni_sources} APPEND PROPERTY
          COMPILE_OPTIONS "-mavx512bw")
      set_source_files_properties(${_avx512bw_vnni_sources} PROPERTIES
          COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")
    endif()
  else()
    set_source_files_properties(${_avx512bw_sources} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512BW=1)
endif()

if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
  target_compile_options(lib_onnx_light_cpu PUBLIC --coverage -O0 -g)
  target_link_options(lib_onnx_light_cpu PUBLIC --coverage)
endif()

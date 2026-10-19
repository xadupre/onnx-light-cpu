# Hardening.cmake
#
# Implements the OpenSSF "Compiler Options Hardening Guide for C and C++"
# (https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html)
# as an opt-in set of compile and link flags applied to every ml_cpu_compute
# target.
#
# Enable with ``-DML_CPU_COMPUTE_HARDENING=ON``. Each flag is probed and silently skipped
# when the active toolchain does not support it.
#
# The macro ``ml_cpu_compute_apply_hardening(<target>)`` is the public entry point.
# It is a no-op when ML_CPU_COMPUTE_HARDENING is OFF.

include_guard(GLOBAL)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag OPTIONAL RESULT_VARIABLE _ml_cpu_compute_have_check_linker_flag)

set(ML_CPU_COMPUTE_HARDENING_COMPILE_OPTIONS "" CACHE INTERNAL "")
set(ML_CPU_COMPUTE_HARDENING_COMPILE_DEFINITIONS "" CACHE INTERNAL "")
set(ML_CPU_COMPUTE_HARDENING_LINK_OPTIONS "" CACHE INTERNAL "")

function(_ml_cpu_compute_try_cxx_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ML_CPU_COMPUTE_HARDENING_CXX_${flag}" cache_var)
  check_cxx_compiler_flag("${flag}" ${cache_var})
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_ml_cpu_compute_try_link_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ML_CPU_COMPUTE_HARDENING_LD_${flag}" cache_var)
  if(_ml_cpu_compute_have_check_linker_flag)
    check_linker_flag(CXX "${flag}" ${cache_var})
  else()
    set(CMAKE_REQUIRED_LINK_OPTIONS_BACKUP "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_cxx_compiler_flag("" ${cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${CMAKE_REQUIRED_LINK_OPTIONS_BACKUP}")
  endif()
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_ml_cpu_compute_normalize_msvc_arch arch out_var)
  string(TOLOWER "${arch}" _arch)
  if(_arch STREQUAL "win32" OR _arch STREQUAL "x86")
    set(${out_var} "x86" PARENT_SCOPE)
  elseif(_arch STREQUAL "x64" OR _arch STREQUAL "amd64" OR _arch STREQUAL "x86_64")
    set(${out_var} "x64" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm")
    set(${out_var} "arm" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm64" OR _arch STREQUAL "aarch64")
    set(${out_var} "arm64" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm64ec")
    set(${out_var} "arm64ec" PARENT_SCOPE)
  else()
    set(${out_var} "${_arch}" PARENT_SCOPE)
  endif()
endfunction()

function(_ml_cpu_compute_get_msvc_spectre_lib_dir out_var)
  if(NOT DEFINED ENV{VCToolsInstallDir} OR "$ENV{VCToolsInstallDir}" STREQUAL "")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_VS_PLATFORM_NAME)
    set(_target_arch "${CMAKE_VS_PLATFORM_NAME}")
  elseif(CMAKE_GENERATOR_PLATFORM)
    set(_target_arch "${CMAKE_GENERATOR_PLATFORM}")
  elseif(DEFINED ENV{VSCMD_ARG_TGT_ARCH} AND NOT "$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "")
    set(_target_arch "$ENV{VSCMD_ARG_TGT_ARCH}")
  elseif(CMAKE_SYSTEM_PROCESSOR)
    set(_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
  else()
    set(_target_arch "x64")
  endif()
  _ml_cpu_compute_normalize_msvc_arch("${_target_arch}" _target_arch)

  file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _vctools_dir)
  set(_spectre_dir "${_vctools_dir}/lib/spectre/${_target_arch}")
  if(IS_DIRECTORY "${_spectre_dir}")
    set(${out_var} "${_spectre_dir}" PARENT_SCOPE)
    return()
  endif()

  if(_target_arch STREQUAL "arm64ec")
    set(_arm64_fallback "${_vctools_dir}/lib/spectre/arm64")
    if(IS_DIRECTORY "${_arm64_fallback}")
      set(${out_var} "${_arm64_fallback}" PARENT_SCOPE)
      return()
    endif()
  endif()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

set(_ml_cpu_compute_hardening_compile_options "")
set(_ml_cpu_compute_hardening_compile_definitions "")
set(_ml_cpu_compute_hardening_link_options "")

if(MSVC)
  set(_msvc_compile_candidates
      /GS
      /guard:cf
      /Qspectre
      /sdl
      /ZH:SHA_256
  )
  foreach(flag IN LISTS _msvc_compile_candidates)
    _ml_cpu_compute_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _ml_cpu_compute_hardening_compile_options "${flag}")
    endif()
  endforeach()

  set(_msvc_link_candidates
      /DYNAMICBASE
      /NXCOMPAT
      /CETCOMPAT
      /guard:cf
  )
  foreach(flag IN LISTS _msvc_link_candidates)
    _ml_cpu_compute_try_link_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _ml_cpu_compute_hardening_link_options "${flag}")
    endif()
  endforeach()

  list(FIND _ml_cpu_compute_hardening_compile_options "/Qspectre" _ml_cpu_compute_qspectre_index)
  if(NOT _ml_cpu_compute_qspectre_index EQUAL -1)
    _ml_cpu_compute_get_msvc_spectre_lib_dir(_ml_cpu_compute_msvc_spectre_lib_dir)
    if(_ml_cpu_compute_msvc_spectre_lib_dir)
      list(APPEND _ml_cpu_compute_hardening_link_options
           "SHELL:/LIBPATH:\"${_ml_cpu_compute_msvc_spectre_lib_dir}\"")
    else()
      message(WARNING
              "ML_CPU_COMPUTE_HARDENING: /Qspectre is enabled but the MSVC Spectre-mitigated "
              "CRT/STL libraries could not be located.")
    endif()
  endif()
else()
  set(_gcc_compile_candidates
      -Wall
      -Wformat
      -Wformat=2
      -Wimplicit-fallthrough
      -Werror=format-security
      -fstack-protector-strong
      -fstack-clash-protection
      -fcf-protection=full
      -fstrict-flex-arrays=3
      -fno-delete-null-pointer-checks
      -fno-strict-overflow
      -fno-strict-aliasing
      -ftrivial-auto-var-init=zero
      -fexceptions
  )
  foreach(flag IN LISTS _gcc_compile_candidates)
    _ml_cpu_compute_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _ml_cpu_compute_hardening_compile_options "${flag}")
    endif()
  endforeach()

  _ml_cpu_compute_try_cxx_flag("-D_FORTIFY_SOURCE=3" _have_fortify3)
  if(_have_fortify3)
    list(APPEND _ml_cpu_compute_hardening_compile_definitions "_FORTIFY_SOURCE=3")
  else()
    _ml_cpu_compute_try_cxx_flag("-D_FORTIFY_SOURCE=2" _have_fortify2)
    if(_have_fortify2)
      list(APPEND _ml_cpu_compute_hardening_compile_definitions "_FORTIFY_SOURCE=2")
    endif()
  endif()
  list(APPEND _ml_cpu_compute_hardening_compile_definitions "_GLIBCXX_ASSERTIONS")

  set(_gcc_link_candidates
      "LINKER:-z,noexecstack"
      "LINKER:-z,relro"
      "LINKER:-z,now"
      "LINKER:--as-needed"
      "LINKER:--no-copy-dt-needed-entries"
  )
  foreach(flag IN LISTS _gcc_link_candidates)
    _ml_cpu_compute_try_link_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _ml_cpu_compute_hardening_link_options "${flag}")
    endif()
  endforeach()
endif()

set(ML_CPU_COMPUTE_HARDENING_COMPILE_OPTIONS "${_ml_cpu_compute_hardening_compile_options}"
    CACHE INTERNAL "Hardening compile options resolved for this toolchain.")
set(ML_CPU_COMPUTE_HARDENING_COMPILE_DEFINITIONS "${_ml_cpu_compute_hardening_compile_definitions}"
    CACHE INTERNAL "Hardening compile definitions resolved for this toolchain.")
set(ML_CPU_COMPUTE_HARDENING_LINK_OPTIONS "${_ml_cpu_compute_hardening_link_options}"
    CACHE INTERNAL "Hardening link options resolved for this toolchain.")

message(STATUS "ML_CPU_COMPUTE_HARDENING: enabled "
        "(compile: ${ML_CPU_COMPUTE_HARDENING_COMPILE_OPTIONS}; "
        "define: ${ML_CPU_COMPUTE_HARDENING_COMPILE_DEFINITIONS}; "
        "link: ${ML_CPU_COMPUTE_HARDENING_LINK_OPTIONS})")

function(ml_cpu_compute_apply_hardening target)
  if(NOT TARGET ${target})
    return()
  endif()
  if(ML_CPU_COMPUTE_HARDENING_COMPILE_OPTIONS)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:${ML_CPU_COMPUTE_HARDENING_COMPILE_OPTIONS}>)
  endif()
  if(ML_CPU_COMPUTE_HARDENING_COMPILE_DEFINITIONS)
    target_compile_definitions(${target} PRIVATE
        ${ML_CPU_COMPUTE_HARDENING_COMPILE_DEFINITIONS})
  endif()
  if(ML_CPU_COMPUTE_HARDENING_LINK_OPTIONS)
    target_link_options(${target} PRIVATE
        ${ML_CPU_COMPUTE_HARDENING_LINK_OPTIONS})
  endif()
endfunction()

# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

if(NOT CONFIG_CMSIS_EXECUTORCH)
  return()
endif()

if(CONFIG_EXECUTORCH)
  message(FATAL_ERROR
    "CMSIS_EXECUTORCH and the upstream EXECUTORCH runtime cannot be enabled together."
  )
endif()

get_filename_component(CMSIS_EXECUTORCH_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(CMSIS_EXECUTORCH_SOURCE_DIR "${ZEPHYR_EXECUTORCH_MODULE_DIR}")
if(NOT EXISTS "${CMSIS_EXECUTORCH_SOURCE_DIR}/version.txt")
  message(FATAL_ERROR
    "CMSIS-ExecuTorch requires the installed executorch Zephyr module. "
    "Add it to the workspace and keep CONFIG_EXECUTORCH=n."
  )
endif()

# Generated artifacts must match the source schemas and operator signatures.
# Normalize Windows checkout line endings before checking their fingerprints.
function(cmsis_executorch_check_input relative_path expected_hash)
  set(input "${CMSIS_EXECUTORCH_SOURCE_DIR}/${relative_path}")
  if(NOT EXISTS "${input}")
    message(FATAL_ERROR "ExecuTorch input is missing: ${input}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
  file(READ "${input}" contents)
  string(REPLACE "\r\n" "\n" contents "${contents}")
  string(SHA256 actual_hash "${contents}")
  if(NOT actual_hash STREQUAL expected_hash)
    message(FATAL_ERROR
      "${relative_path} differs from the schemas used for CMSIS-ExecuTorch's "
      "pre-generated artifacts. Use a matching checkout or refresh the artifacts "
      "separately; the firmware build does not run code generators."
    )
  endif()
endfunction()
include("${CMAKE_CURRENT_LIST_DIR}/compatibility.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMSIS_EXECUTORCH_SOURCE_DIR}/version.txt")
file(STRINGS "${CMSIS_EXECUTORCH_SOURCE_DIR}/version.txt" executorch_version LIMIT_COUNT 1)
if(NOT executorch_version STREQUAL CMSIS_EXECUTORCH_EXPECTED_VERSION)
  message(FATAL_ERROR
    "CMSIS-ExecuTorch requires ExecuTorch ${CMSIS_EXECUTORCH_EXPECTED_VERSION}; "
    "the installed module reports ${executorch_version}."
  )
endif()

set(flatbuffers_include "${CMSIS_EXECUTORCH_SOURCE_DIR}/third-party/flatbuffers/include")
if(NOT EXISTS "${flatbuffers_include}/flatbuffers/flatbuffers.h")
  message(FATAL_ERROR
    "The ExecuTorch checkout needs its pinned third-party/flatbuffers submodule "
    "(headers only). Initialize it before building the application."
  )
endif()

# Upstream includes use <executorch/...>; west installs this checkout as executorch.
get_filename_component(executorch_parent "${CMSIS_EXECUTORCH_SOURCE_DIR}" DIRECTORY)
get_filename_component(executorch_directory_name "${CMSIS_EXECUTORCH_SOURCE_DIR}" NAME)
if(NOT executorch_directory_name STREQUAL "executorch")
  message(FATAL_ERROR "The upstream checkout directory must be named executorch for its include paths.")
endif()

set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/generated")
include("${CMAKE_CURRENT_LIST_DIR}/sources.cmake")
set(CMSIS_EXECUTORCH_OPERATOR_SOURCES "")
set(CMSIS_EXECUTORCH_SELECTED_OPERATORS "")
set(CMSIS_EXECUTORCH_KERNEL_COUNT 0)
function(cmsis_executorch_operator name kernel_count)
  if(CONFIG_CMSIS_EXECUTORCH_OP_${name})
    foreach(source IN LISTS ARGN)
      list(APPEND CMSIS_EXECUTORCH_OPERATOR_SOURCES "${CMSIS_EXECUTORCH_SOURCE_DIR}/${source}")
    endforeach()
    list(APPEND CMSIS_EXECUTORCH_SELECTED_OPERATORS "${name}")
    math(EXPR CMSIS_EXECUTORCH_KERNEL_COUNT "${CMSIS_EXECUTORCH_KERNEL_COUNT} + ${kernel_count}")
    set(CMSIS_EXECUTORCH_OPERATOR_SOURCES "${CMSIS_EXECUTORCH_OPERATOR_SOURCES}" PARENT_SCOPE)
    set(CMSIS_EXECUTORCH_SELECTED_OPERATORS "${CMSIS_EXECUTORCH_SELECTED_OPERATORS}" PARENT_SCOPE)
    set(CMSIS_EXECUTORCH_KERNEL_COUNT "${CMSIS_EXECUTORCH_KERNEL_COUNT}" PARENT_SCOPE)
  endif()
endfunction()
include("${CMAKE_CURRENT_LIST_DIR}/operators.cmake")

if(CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION)
  file(MAKE_DIRECTORY "${generated_directory}")
  # Only create the translation unit; declarations/registration are pre-generated.
  file(CONFIGURE OUTPUT "${generated_directory}/register_kernels.cpp"
    CONTENT "// Registration is pre-generated; no host code generator is run.\n#include <cmsis_executorch/kernels.h>\n"
  )
endif()
set(module_sources ${CMSIS_EXECUTORCH_OPERATOR_SOURCES})
foreach(component IN LISTS CMSIS_EXECUTORCH_COMPONENTS)
  if(CONFIG_CMSIS_EXECUTORCH_${component})
    list(APPEND module_sources ${CMSIS_EXECUTORCH_${component}_SOURCES})
  endif()
endforeach()
if(CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE)
  list(APPEND module_sources "${CMSIS_EXECUTORCH_MODULE_DIR}/src/arm_embedded_module.cpp")
endif()
foreach(source IN LISTS module_sources)
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "ExecuTorch source is missing: ${source}")
  endif()
endforeach()

# These definitions also apply to application code that includes runtime headers.
zephyr_include_directories(
  "${CMAKE_CURRENT_LIST_DIR}/include"
  "${executorch_parent}"
  "${CMSIS_EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type/c10"
  "${flatbuffers_include}"
)
if(CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE)
  zephyr_include_directories("${CMSIS_EXECUTORCH_MODULE_DIR}/src")
endif()
zephyr_compile_definitions(
  C10_USING_CUSTOM_GENERATED_MACROS
  FLATBUFFERS_MAX_ALIGNMENT=1024
  FLATBUFFERS_LOCALE_INDEPENDENT=0
  ET_LOG_ENABLED=$<BOOL:${CONFIG_CMSIS_EXECUTORCH_LOG_ENABLED}>
  ET_MIN_LOG_LEVEL=${CONFIG_CMSIS_EXECUTORCH_MIN_LOG_LEVEL}
  ET_ENABLE_PROGRAM_VERIFICATION=$<BOOL:${CONFIG_CMSIS_EXECUTORCH_PROGRAM_VERIFICATION}>
  ET_ENABLE_ENUM_STRINGS=$<BOOL:${CONFIG_CMSIS_EXECUTORCH_ENUM_STRINGS}>
  ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=$<BOOL:${CONFIG_CMSIS_EXECUTORCH_DEPRECATED_CONSTANT_BUFFER}>
  ET_MAX_INSTRUCTIONS=${CONFIG_CMSIS_EXECUTORCH_MAX_INSTRUCTIONS}
)
# These switches use #ifdef/defined in upstream headers: defining them to 0
# would still enable the feature. Export them only when selected, including
# to application code so EventTracer and profiling headers agree with runtime.
if(CONFIG_CMSIS_EXECUTORCH_EVENT_TRACER)
  zephyr_compile_definitions(ET_EVENT_TRACER_ENABLED)
endif()
if(CONFIG_CMSIS_EXECUTORCH_OP_REGISTRY_DEBUG)
  zephyr_compile_definitions(ET_OP_REGISTRY_DEBUG)
endif()
if(CONFIG_CMSIS_EXECUTORCH_PROFILING)
  zephyr_compile_definitions(
    PROFILING_ENABLED
    MAX_PROFILE_EVENTS=${CONFIG_CMSIS_EXECUTORCH_MAX_PROFILE_EVENTS}
    MAX_MEM_PROFILE_EVENTS=${CONFIG_CMSIS_EXECUTORCH_MAX_MEM_PROFILE_EVENTS}
    MEM_PROFILE_MAX_ALLOCATORS=${CONFIG_CMSIS_EXECUTORCH_MEM_PROFILE_MAX_ALLOCATORS}
    MAX_PROFILE_BLOCKS=${CONFIG_CMSIS_EXECUTORCH_MAX_PROFILE_BLOCKS}
  )
endif()
if(CONFIG_CMSIS_EXECUTORCH_ETHOS_U_PER_DELEGATE_PROFILING)
  zephyr_compile_definitions(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
endif()
if(CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U)
  zephyr_compile_definitions(EXECUTORCH_BUILD_ARM_BAREMETAL=1 ET_USE_ETHOS_U_BACKEND=1)
endif()

if(NOT module_sources)
  return()
endif()

# Zephyr whole-archives its module libraries, retaining the registration constructor.
zephyr_library_named(cmsis_executorch)
zephyr_library_sources(${module_sources})
math(EXPR kernel_capacity "${CMSIS_EXECUTORCH_KERNEL_COUNT} + ${CONFIG_CMSIS_EXECUTORCH_EXTRA_KERNEL_SLOTS}")
if(kernel_capacity LESS 1)
  set(kernel_capacity 1)
endif()
zephyr_library_compile_definitions(MAX_KERNEL_NUM=${kernel_capacity})
list(LENGTH CMSIS_EXECUTORCH_SELECTED_OPERATORS operator_count)
message(STATUS "CMSIS-ExecuTorch: ${operator_count} operator families, ${CMSIS_EXECUTORCH_KERNEL_COUNT} registrations")

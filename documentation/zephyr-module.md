# CMSIS-ExecuTorch in Zephyr

This repository is the `cmsis-executorch` Zephyr module. It compiles runtime
and operator sources from the **installed `executorch` module**, without
entering that module's upstream CMake build. The upstream module remains
installed with `CONFIG_EXECUTORCH=n`.

The additions supplied here are:

- Pre-generated `program_generated.h` and `scalar_type_generated.h`.
- Pre-generated kernel declarations and registration in
  `zephyr/include/cmsis_executorch/kernels.h`.
- Kconfig and CMake metadata for selecting every pack component and its sources.
- The existing `src/arm_embedded_module.cpp` and `.hpp`, used in place.

Runtime, kernel, c10, FlatBuffers, and platform implementations are read from
the installed ExecuTorch checkout. This module does not bundle them. Existing
CMSIS sources and solution files are unchanged. The firmware build runs no
`flatc`, `flatcc`, registration generator, host-tool build, or tool downloader.
Zephyr still uses its normal Python build infrastructure.

## Installation and versions

Keep this repository alongside the application or install it as another west
project. The workspace must already contain the `executorch` module, normally
at `modules/lib/executorch`, and its pinned `third-party/flatbuffers` submodule
must be populated. Only FlatBuffers headers are consumed.

The supported ExecuTorch version is recorded in `zephyr/compatibility.cmake`.
The generated schema headers assert their required FlatBuffers version.
CMake checks the reported ExecuTorch version and fingerprints of the schemas
and operator YAML, ignoring Windows CRLF versus LF differences. A mismatch
requires refreshing the artifacts separately before building.

For a checkout outside the west manifest, add its repository root to the
application's `CMakeLists.txt` before `find_package(Zephyr)`:

```cmake
cmake_minimum_required(VERSION 3.20.0)

list(APPEND EXTRA_ZEPHYR_MODULES "C:/path/to/CMSIS-Executorch")
list(APPEND EXTRA_CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/operators.conf")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(my_app)
target_sources(app PRIVATE src/main.cpp)
```

During firmware configuration, the installed upstream checkout is located
automatically through `ZEPHYR_EXECUTORCH_MODULE_DIR`; no CMSIS pack installation or pack-root
environment variable is required. Its directory must be named `executorch`
because upstream headers use the `<executorch/...>` include prefix.

These application CMake settings also apply when csolution invokes west.
There is no additional generator executable to discover on `PATH`.

## West AI-layer command

This repository provides a `west ailayer` extension in
[`zephyr/ailayer.py`](../zephyr/ailayer.py). Register it by adding
`west-commands` to this project's existing entry in your workspace's `west.yml`:

```yaml
manifest:
  projects:
    - name: cmsis-executorch
      # Keep the project's existing URL, revision and path settings.
      west-commands: zephyr/west-commands.yml
```

If this repository is itself the manifest repository, put the same
`west-commands` field under `manifest.self` instead. This is
[west extension registration](https://docs.zephyrproject.org/latest/develop/west/extensions.html),
separate from `zephyr/module.yml`; `EXTRA_ZEPHYR_MODULES` alone does not register
west commands.

From the application directory, run:

```shell
west ailayer
```

The command uses the only `*.cbuild-mlops.yml` in the current working directory.
If none or several exist, it reports the problem and asks for an explicit path:

```shell
west ailayer path/to/solution.cbuild-mlops.yml
```

The output is always `ai_layer/` beside the input file. For example, an input
in `applications/my_app/` produces:

```text
applications/my_app/ai_layer/
  CMakeLists.txt
  model.pte
  model_pte.c
  model_pte.h
  components.conf
  operators.conf
```

Add the generated directory before `find_package(Zephyr)` in the application's
`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20.0)
add_subdirectory(ai_layer)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(my_app)
target_sources(app PRIVATE src/main.cpp)
```

The generated CMake file appends both fragments to `EXTRA_CONF_FILE` before
Kconfig runs, then links the model source and include directory into `app`
after Zephyr creates that target. Existing extra fragments are preserved.
Keep C++17 or newer and the board's Ethos-U driver/devicetree configuration in
the application. The generated component fragment enables the CMSIS-ExecuTorch
runtime, utilities, registration, Ethos-U backend, and disables ExecuTorch
logging to match the CMSIS output. It also enables CMSIS-NN when a selected
Cortex-M operator requires it.

The wrapper imports `create_ai_layer()` from
[`create_ai_layer.py`](../create_ai_layer.py) and selects the Zephyr backend.
The two output backends live in [`ai_layer_backends/`](../ai_layer_backends/).
Both consume the same `cbuild-mlops` input, NPU/Vela settings, model parameters,
and shared export result. CMSIS uses `model.clayer` (default:
`ai_layer/ai_layer.clayer.yml`); Zephyr writes `ai_layer/` beside the input and
ignores that CMSIS-specific output path. Model parameters are optional; the
model's defaults are used when no `model:` node is present. The exporter loads
`model/model.py` beside the input when available, otherwise the repository's
starter model. It must provide `get_model()` and `get_calibration_inputs()`.

Zephyr generation selects exact CPU operator overloads from all exported
methods using this module's registration catalog. Unsupported operators cause
an error before output files are replaced. Zephyr generation needs no installed
CMSIS pack or `.cbuild-pack.yml`; CMSIS generation still uses both. Generation
is an explicit command and is not invoked during firmware builds.

Library callers can use the same function:

```python
from create_ai_layer import create_ai_layer

cmsis_layer = create_ai_layer("path/to/solution.cbuild-mlops.yml")
zephyr_cmake = create_ai_layer("path/to/solution.cbuild-mlops.yml", backend="zephyr")
```

Importing the module performs no generation and does not load the ML packages.
The function returns the generated `.clayer.yml` or `CMakeLists.txt` path and raises exceptions on
failure. Library calls and `west ailayer` use the caller's Python environment,
which must contain the export dependencies in `requirements.txt`. Only the
standalone `python create_ai_layer.py ...` CLI automatically relaunches itself
in this repository's `.venv`.

## Select the application's operators

In `prj.conf`, enable C++ and this module:

```ini
CONFIG_CPP=y
CONFIG_STD_CPP17=y
CONFIG_EXECUTORCH=n
CONFIG_CMSIS_EXECUTORCH=y
```

In the application's `operators.conf`, select the required families:

```ini
CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_QUANTIZE=y
CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE=y
```

With this file, the only operator implementation files compiled are upstream
`kernels/quantized/cpu/op_quantize.cpp` and `op_dequantize.cpp`. They provide
eight registered variants between them. Shared runtime and kernel utilities
are also compiled. The other operator implementation files and registration
entries are excluded before compilation, independently of linker pruning.

No operator is enabled by default. Configuration options cover the pack's
portable, quantized, and Cortex-M operator families. They appear under
CMSIS-ExecuTorch in `menuconfig`; the complete symbol list is in
[`zephyr/Kconfig.operators`](../zephyr/Kconfig.operators).

For example, add portable addition with:

```ini
CONFIG_CMSIS_EXECUTORCH_OP_PORTABLE_ADD=y
```

The unit of selection is one pack **operator component**, including all its
registered overloads and variants. Quantize therefore includes its per-tensor,
per-channel, and per-token variants. This first integration does not select
individual overloads or data types within a component.

Cortex-M options require `CONFIG_CPU_CORTEX_M`. Operators that call CMSIS-NN
also require the CMSIS-NN module and `CONFIG_CMSIS_NN=y`; their options select
the needed CMSIS-NN function groups. No CMSIS-NN dependency is introduced by
the portable quantize/dequantize example above.

## Pack components

Every component in PyTorch::ExecuTorch@1.5.1 has a Kconfig option: 199 operator
components and these five shared components. The latter are declared in
[`zephyr/Kconfig.components`](../zephyr/Kconfig.components).

| Option | Sources and dependencies | Default |
| --- | --- | --- |
| `CONFIG_CMSIS_EXECUTORCH_RUNTIME` | Core runtime and shared embedding helper; uses the Zephyr PAL | Enabled |
| `CONFIG_CMSIS_EXECUTORCH_KERNEL_UTILS` | Portable kernel utilities and patterns; selects Runtime | Selected by operators and Kernel Registration |
| `CONFIG_CMSIS_EXECUTORCH_EXTENSION_TENSOR` | `tensor_ptr.cpp` and `tensor_ptr_maker.cpp`; selects Runtime | Disabled |
| `CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION` | Pre-generated registration for the selected operators; selects Kernel Utils | Enabled |
| `CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U` | Ethos-U backend; requires Cortex-M and `CONFIG_ETHOS_U`; selects Runtime | Disabled |

For an Ethos-U delegated model, configure the board's Zephyr NPU driver and add:

```ini
CONFIG_ETHOS_U=y
CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y
```

The backend compiles upstream `backends/arm/runtime/EthosUBackend.cpp`,
`EthosUBackend_Cortex_M.cpp`, `EthosUBackend_IoMemcpy.cpp`, and `VelaBinStream.cpp`.
It also sets the pack's `EXECUTORCH_BUILD_ARM_BAREMETAL=1` and
`ET_USE_ETHOS_U_BACKEND=1` definitions. Driver initialization, devicetree and
the NPU configuration remain part of the board/application setup.

Enable `CONFIG_CMSIS_EXECUTORCH_EXTENSION_TENSOR` when using `TensorPtr` or
`make_tensor_ptr`. Applications supplying their own kernel registration can
disable `CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION`; selected operator
implementations and their utilities still compile. Kernel Utils can also be
enabled explicitly for application-defined kernels. The EmbeddedModule facade
selects Runtime as well.

## Generated operator configuration

`west ailayer` runs **before Zephyr configuration** and emits a plain Kconfig
fragment in the format shown above. The generated CMake file merges it using
`EXTRA_CONF_FILE`, alongside `prj.conf`. Zephyr owns the merged configuration
and applies the same operator settings to both CMake source selection and the
pre-generated registration header. There is no second operator list to keep
in sync.

The generator maps each required operator to its registered component and
accounts for all methods in the exported program. The upstream YAML defines
operator schemas and implementation names; `zephyr/operators.cmake` maps the
selectable family names to sources.
Delegated operations require their backend separately; selecting CPU kernels
does not enable an accelerator backend.

When a model changes, run `west ailayer` again and reconfigure the application.
The generated fragments are replaced so stale operator selections disappear.
Keep manually added operators in a separate fragment if they must survive
model regeneration. A standalone analyser for pre-existing PTE files is not
provided.

## EmbeddedModule and runtime settings

`CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE=y` is the default. Include
`arm_embedded_module.hpp` from application C++ and supply the data loader and
allocators required by the existing class. Heap and memory pool sizing remain
application responsibilities.

The upstream `runtime/platform/default/zephyr.cpp` provides the PAL. Runtime
settings live in [`zephyr/Kconfig.executorch`](../zephyr/Kconfig.executorch),
under the "ExecuTorch runtime settings" menu. CMake maps these local symbols to
the original C/C++ macros, for both the runtime and application code including
its headers:

| `CONFIG_CMSIS_EXECUTORCH_` suffix | Compiler definition | Default |
| --- | --- | --- |
| `LOG_ENABLED` | `ET_LOG_ENABLED=0/1` | Enabled |
| `LOG_LEVEL_DEBUG/INFO/ERROR/FATAL` (choice) | `ET_MIN_LOG_LEVEL=Debug/Info/Error/Fatal` | Info |
| `PROGRAM_VERIFICATION` | `ET_ENABLE_PROGRAM_VERIFICATION=0/1` | Enabled |
| `ENUM_STRINGS` | `ET_ENABLE_ENUM_STRINGS=0/1` | Enabled |
| `DEPRECATED_CONSTANT_BUFFER` | `ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0/1` | Enabled, matching the pack source default |
| `MAX_INSTRUCTIONS` | `ET_MAX_INSTRUCTIONS` | 10000000 |
| `EVENT_TRACER` | `ET_EVENT_TRACER_ENABLED` | Undefined (disabled) |
| `OP_REGISTRY_DEBUG` | `ET_OP_REGISTRY_DEBUG` | Undefined (disabled) |
| `PROFILING` | `PROFILING_ENABLED` | Undefined (disabled) |
| `MAX_PROFILE_EVENTS` | `MAX_PROFILE_EVENTS` | 1024 |
| `MAX_MEM_PROFILE_EVENTS` | `MAX_MEM_PROFILE_EVENTS` | 1024 |
| `MEM_PROFILE_MAX_ALLOCATORS` | `MEM_PROFILE_MAX_ALLOCATORS` | 32 |
| `MAX_PROFILE_BLOCKS` | `MAX_PROFILE_BLOCKS` | 2 |
| `ETHOS_U_PER_DELEGATE_PROFILING` | `ET_ARM_ETHOSU_PER_DELEGATE_PROFILING` | Undefined (disabled) |

For example, compile only error and fatal diagnostics with:

```ini
CONFIG_CMSIS_EXECUTORCH_LOG_ENABLED=y
CONFIG_CMSIS_EXECUTORCH_LOG_LEVEL_ERROR=y
```

The presence-based switches are defined only when enabled: defining
`ET_EVENT_TRACER_ENABLED=0`, for example, would still enable upstream `#ifdef`
branches. Event tracing supplies hooks for an application-provided
`executorch::runtime::EventTracer`. ETDump and its FlatCC code generation are
not included. The built-in profiler is independent of EventTracer; its four
buffer settings appear when `CONFIG_CMSIS_EXECUTORCH_PROFILING=y`.

### Relationship to upstream Zephyr Kconfig

The installed ExecuTorch module's `zephyr/Kconfig` is already included by
Zephyr, but places its feature options inside `if EXECUTORCH`. With
`CONFIG_EXECUTORCH=n`, those definitions inherit a disabled dependency. Enabling
`CONFIG_EXECUTORCH` would also start its upstream CMake build alongside ours.
An additional `source` statement cannot remove the inherited dependency or
rename those symbols.

Local `CMSIS_EXECUTORCH_*` settings therefore provide the controls for this
direct-source integration. Kconfig permits additional definitions of the same
symbol, but their dependencies are combined with OR; this would still require
maintaining local definitions and would couple us to the native integration.
See [Zephyr's multiple-definition rules](https://docs.zephyrproject.org/latest/build/kconfig/setting.html#multiple-symbol-definitions).

Sharing the definitions cleanly would require an upstream refactor: put common
feature options under a runtime-configuration symbol usable by both builds,
while leaving the native CMake build gated on `EXECUTORCH`. Until then, review
`Kconfig.executorch` and its compiler-definition mappings when upgrading the pack.
Per-component operator selection replaces upstream's all-portable-operators
switch. Compiler size optimization remains governed by Zephyr's compiler
configuration. Pack-required c10 and FlatBuffers definitions remain fixed in
CMake to match the generated headers.

The kernel registry is sized for the selected registrations. Applications
that register additional custom kernels can reserve slots with
`CONFIG_CMSIS_EXECUTORCH_EXTRA_KERNEL_SLOTS`.

## Updating generated artifacts

Run the released [`update_zephyr.py`](../update_zephyr.py) script when updating
ExecuTorch. Supply both the matching upstream checkout and the original
PyTorch::ExecuTorch CMSIS pack explicitly:

```powershell
python update_zephyr.py `
  --executorch-dir "C:/path/to/zephyr-repo/modules/lib/executorch" `
  --pack "C:/ctools_pack/PyTorch/ExecuTorch/1.5.1"
```

`--pack` accepts an installed/unpacked pack directory or a downloaded `.pack`
archive. `--executorch-dir` points to the upstream checkout root, containing
`version.txt`, `schema/`, `kernels/`, and the populated FlatBuffers submodule.
The updater does not discover this path through west, `CMSIS_PACK_ROOT`, or
other environment variables. Relative paths are resolved from the working
directory. Its default output is this repository's `zephyr/` directory,
regardless of the working directory; `--output-dir` can override it.

The script uses Python 3.10 or newer and only the standard library. It works
with Windows Python, including a Python environment managed by uv. It does
not invoke bash, west, `flatc`, `flatcc`, or upstream generators. The pack
already supplies the generated schema headers and kernel registration.

The updater reads the version and all components from the pack manifest.
Before writing, it checks that the supplied checkout has the same version,
schemas, operator YAML, and component source contents, and
that its FlatBuffers version matches the generated schema headers. The pack's
minimal PAL is replaced by the upstream Zephyr PAL. It then refreshes:

- `compatibility.cmake`: version and schema/YAML fingerprints.
- `sources.cmake` and `Kconfig.components`: shared components and their sources.
- `operators.cmake` and `Kconfig.operators`: selectable operators and counts.
- Both generated schema headers and `include/cmsis_executorch/kernels.h`,
  adapting the pack's registration guards to Zephyr Kconfig symbols.

Existing CMSIS sources and the supplied upstream checkout and pack are left
unchanged. Run the same command with `--check` to validate without writing:
exit code 0 means current, 1 means artifacts need refreshing, and 2 means an
input is missing or incompatible. Input validation completes before any
artifact is written.

Unmapped or missing pack components, new pack layouts, conditional registrations,
or previously unknown CMSIS-NN operator mappings require review of the updater rather than guessing build
settings. A successful refresh checks the known inputs; application compilation
and execution must still be tested against the new release. Run the checks
below and commit the refreshed artifacts with the release, so consumers can
build without a CMSIS pack installation or code generation tools.

The schema and registration headers use ExecuTorch's BSD-3-Clause license;
see [`src/LICENSE-ExecuTorch`](../src/LICENSE-ExecuTorch).

## Configuration checks

Run the AI-layer backend and west discovery tests using the Zephyr Python
environment (PyYAML and west required). They use fixture exports and do not
load ML packages or run model export:

```powershell
python tests/zephyr/test_ai_layer.py
```

Run the updater tests with any Python 3.10+ environment:

```powershell
python tests/zephyr/test_update_zephyr.py
```

Run the selection checks with the existing Zephyr virtual-environment Python:

```powershell
C:/path/to/zephyr-repo/.venv/Scripts/python.exe tests/zephyr/check_module.py `
  --zephyr-base C:/path/to/zephyr-repo/zephyr `
  --executorch-dir C:/path/to/zephyr-repo/modules/lib/executorch `
  --pack C:/ctools_pack/PyTorch/ExecuTorch/1.5.1
```

These checks exercise Zephyr module discovery, Kconfig, and CMake source
selection, including paths with spaces, reconfiguration, all component options,
backend dependencies, and mismatched generated artifacts. The optional `--pack`
argument audits every PDSC component and source against the selected build
sources. These checks do not compile or run firmware.

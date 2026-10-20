"""Check module discovery, Kconfig selection, and CMake source exclusion.

Run with the existing Zephyr virtual-environment Python. No board, firmware
build, model exporter, or CMSIS pack installation is used by these checks.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zephyr-base', type=Path, required=True)
parser.add_argument('--executorch-dir', type=Path, required=True)
parser.add_argument('--pack', type=Path, help='Optional installed pack or .pack archive for an independent component coverage audit')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
ZEPHYR = args.zephyr_base.resolve()
EXECUTORCH = args.executorch_dir.resolve()
sys.path.insert(0, str(ZEPHYR / 'scripts/kconfig'))
import kconfiglib  # noqa: E402


class ModuleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='module check ', dir=ROOT / 'tmp')
        cls.work = Path(cls.temporary.name).resolve()
        assert cls.work.is_relative_to((ROOT / 'tmp').resolve())
        cls.module = cls.work / 'CMSIS ExecuTorch'
        shutil.copytree(ROOT / 'zephyr', cls.module / 'zephyr')
        (cls.module / 'src').mkdir()
        for name in ['arm_embedded_module.cpp', 'arm_embedded_module.hpp']:
            shutil.copy2(ROOT / 'src' / name, cls.module / 'src' / name)
        cls.fixture = cls.work / 'fixture'
        cls.fixture.mkdir()
        cls.kconfig = cls.fixture / 'Kconfig'
        cls.kconfig.write_text('''
mainmenu "CMSIS-ExecuTorch integration checks"
config CPP
    bool "C++"
    default y
config STD_CPP_VERSION
    int
    default 201703
config REQUIRES_FULL_LIBCPP
    bool
config CPU_CORTEX_M
    bool "Cortex-M"
    default y
config CMSIS_DSP
    bool
config MULTITHREADING
    bool
    default y
config LOG
    bool
source "''' + (ZEPHYR / 'modules/hal_ethos_u/Kconfig').as_posix() + '''"
source "''' + (ZEPHYR / 'modules/cmsis-nn/Kconfig').as_posix() + '''"
source "''' + (EXECUTORCH / 'zephyr/Kconfig').as_posix() + '''"
source "''' + (cls.module / 'zephyr/Kconfig').as_posix() + '''"
''', encoding='utf-8')
        (cls.fixture / 'CMakeLists.txt').write_text('''
cmake_minimum_required(VERSION 3.20)
project(module_selection NONE)
include("${TEST_CONFIGURATION}")
function(zephyr_include_directories)
  set_property(GLOBAL APPEND PROPERTY module_includes ${ARGN})
endfunction()
function(zephyr_compile_definitions)
  set_property(GLOBAL APPEND PROPERTY module_definitions ${ARGN})
  set_property(GLOBAL APPEND PROPERTY app_definitions ${ARGN})
endfunction()
function(zephyr_library_named name)
  set_property(GLOBAL PROPERTY module_library "${name}")
endfunction()
function(zephyr_library_sources source)
  set_property(GLOBAL APPEND PROPERTY module_sources ${source} ${ARGN})
endfunction()
function(zephyr_library_compile_definitions)
  set_property(GLOBAL APPEND PROPERTY module_definitions ${ARGN})
endfunction()
add_subdirectory("${TEST_MODULE}/zephyr" module)
get_property(sources GLOBAL PROPERTY module_sources)
get_property(definitions GLOBAL PROPERTY module_definitions)
get_property(includes GLOBAL PROPERTY module_includes)
get_property(library GLOBAL PROPERTY module_library)
get_property(app_definitions GLOBAL PROPERTY app_definitions)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sources.txt" CONTENT "${sources}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/definitions.txt" CONTENT "${definitions}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/includes.txt" CONTENT "${includes}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/library.txt" CONTENT "${library}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/app_definitions.txt" CONTENT "${app_definitions}")
''', encoding='utf-8')
        cls.registration = (ROOT / 'zephyr/include/cmsis_executorch/kernels.h').read_text(encoding='utf-8')
        cls.catalog = [(name, int(count), paths.split()) for name, count, paths in re.findall(
            r'cmsis_executorch_operator\((\w+) (\d+)\s+([^)]+)\)',
            (ROOT / 'zephyr/operators.cmake').read_text(encoding='utf-8'),
        )]
        cls.counts = {name: count for name, count, _ in cls.catalog}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, name, families=(), enabled=True, extra='', expected_success=True):
        conf = self.fixture / (name + '.conf')
        conf.write_text(
            'CONFIG_CMSIS_EXECUTORCH=' + ('y' if enabled else 'n') + '\n'
            + ''.join('CONFIG_CMSIS_EXECUTORCH_OP_' + family + '=y\n' for family in families)
            + extra,
            encoding='utf-8',
        )
        config = kconfiglib.Kconfig(str(self.kconfig), warn_to_stderr=False)
        config.load_config(str(conf))
        self.assertFalse(config.warnings, '\n'.join(config.warnings))
        cmake_conf = self.fixture / (name + '.cmake')
        lines = [f'set(ZEPHYR_EXECUTORCH_MODULE_DIR "{EXECUTORCH.as_posix()}")']
        for symbol in config.unique_defined_syms:
            if symbol.type in (kconfiglib.BOOL, kconfiglib.TRISTATE, kconfiglib.INT, kconfiglib.HEX, kconfiglib.STRING):
                value = symbol.str_value
                if value == 'n':
                    value = 'OFF'
                lines.append(f'set(CONFIG_{symbol.name} "{value}")')
        cmake_conf.write_text('\n'.join(lines), encoding='utf-8')
        build = self.work / name
        result = subprocess.run([
            'cmake', '-S', str(self.fixture), '-B', str(build),
            '-DTEST_MODULE=' + str(self.module), '-DTEST_CONFIGURATION=' + str(cmake_conf),
        ], text=True, capture_output=True)
        if expected_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
            return result.stdout + result.stderr
        sources = (build / 'sources.txt').read_text().split(';')
        return config, build, {Path(source) for source in sources if source}

    def test_module_discovery(self):
        output = self.work / 'modules.cmake'
        result = subprocess.run([
            sys.executable, str(ZEPHYR / 'scripts/zephyr_module.py'),
            '--zephyr-base', str(ZEPHYR), '--modules', str(EXECUTORCH),
            '--extra-modules', str(self.module), '--cmake-out', str(output),
        ], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = output.read_text()
        self.assertIn('cmsis-executorch', metadata)
        self.assertLess(metadata.index('"executorch"'), metadata.index('"cmsis-executorch"'))

    def test_quantize_dequantize_only_and_reconfiguration(self):
        _, build, sources = self.configure('selection', ['QUANTIZED_QUANTIZE', 'QUANTIZED_DEQUANTIZE'])
        self.assertEqual(
            {p.name for p in sources if p.name.startswith('op_')},
            {'op_quantize.cpp', 'op_dequantize.cpp'},
        )
        self.assertIn(EXECUTORCH / 'runtime/platform/default/zephyr.cpp', sources)
        self.assertNotIn(EXECUTORCH / 'runtime/platform/default/minimal.cpp', sources)
        self.assertIn(self.module / 'src/arm_embedded_module.cpp', sources)
        count = self.counts['QUANTIZED_QUANTIZE'] + self.counts['QUANTIZED_DEQUANTIZE']
        self.assertIn(f'MAX_KERNEL_NUM={count}', (build / 'definitions.txt').read_text())
        self.assertEqual(
            (build / 'module/generated/register_kernels.cpp').read_text().count('#include'), 1,
        )
        _, build, sources = self.configure('selection', ['QUANTIZED_DEQUANTIZE'])
        self.assertEqual({p.name for p in sources if p.name.startswith('op_')}, {'op_dequantize.cpp'})
        self.assertIn(f'MAX_KERNEL_NUM={self.counts["QUANTIZED_DEQUANTIZE"]}', (build / 'definitions.txt').read_text())

    def test_no_operators_by_default(self):
        config, build, sources = self.configure('no_operators')
        self.assertFalse(any(p.name.startswith('op_') for p in sources))
        self.assertTrue(any('/kernels/portable/cpu/util/' in p.as_posix() for p in sources))
        self.assertEqual(config.syms['CMSIS_EXECUTORCH_KERNEL_REGISTRATION'].str_value, 'y')
        self.assertFalse(any('/extension/tensor/' in p.as_posix() or '/backends/arm/' in p.as_posix() for p in sources))
        self.assertIn('MAX_KERNEL_NUM=1', (build / 'definitions.txt').read_text())
        for name, _, _ in self.catalog:
            self.assertEqual(config.syms['CMSIS_EXECUTORCH_OP_' + name].str_value, 'n')

    def test_portable_selection_and_embedding_support(self):
        _, _, sources = self.configure('portable', ['PORTABLE_ADD', 'QUANTIZED_EMBEDDING2B'])
        self.assertEqual({p.name for p in sources if p.name.startswith('op_')}, {'op_add.cpp', 'op_embedding2b.cpp'})
        self.assertIn(EXECUTORCH / 'kernels/quantized/cpu/embeddingxb.cpp', sources)

    def test_component_with_multiple_sources(self):
        catalog = self.module / 'zephyr/operators.cmake'
        original = catalog.read_text()
        catalog.write_text(original.replace(
            'kernels/quantized/cpu/op_quantize.cpp)',
            'kernels/quantized/cpu/op_quantize.cpp\n  kernels/portable/cpu/op_add.cpp)',
        ))
        try:
            _, _, sources = self.configure('multiple_sources', ['QUANTIZED_QUANTIZE'])
            self.assertEqual({p.name for p in sources if p.name.startswith('op_')}, {'op_quantize.cpp', 'op_add.cpp'})
        finally:
            catalog.write_text(original)

    def test_all_operator_sources_and_registration_agree(self):
        config, build, sources = self.configure(
            'all_operators', [name for name, _, _ in self.catalog], extra='CONFIG_CMSIS_NN=y\n',
        )
        expected = {EXECUTORCH / source for _, _, paths in self.catalog for source in paths}
        self.assertTrue(expected.issubset(sources))
        self.assertEqual({p for p in sources if p.name.startswith('op_')}, {p for p in expected if p.name.startswith('op_')})
        self.assertTrue(self.catalog)
        self.assertIn(f'MAX_KERNEL_NUM={sum(self.counts.values())}', (build / 'definitions.txt').read_text())
        for name, count, paths in self.catalog:
            self.assertEqual(config.syms['CMSIS_EXECUTORCH_OP_' + name].str_value, 'y', name)
            blocks = re.findall(
                r'#ifdef CONFIG_CMSIS_EXECUTORCH_OP_' + name + r'\n(?=\s*Kernel\()(.*?)#endif // CONFIG_CMSIS_EXECUTORCH_OP_' + name + r'\b',
                self.registration, re.S,
            )
            self.assertEqual(sum(len(re.findall(r'^Kernel\(', block, re.M)) for block in blocks), count, paths)

    def test_disabled_module_and_optional_facade(self):
        _, build, sources = self.configure('disabled', enabled=False)
        self.assertFalse(sources)
        self.assertEqual((build / 'library.txt').read_text(), '')
        _, build, sources = self.configure('no_facade', extra='CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE=n\nCONFIG_CMSIS_EXECUTORCH_EXTRA_KERNEL_SLOTS=3\n')
        self.assertNotIn(self.module / 'src/arm_embedded_module.cpp', sources)
        self.assertIn('MAX_KERNEL_NUM=3', (build / 'definitions.txt').read_text())

    def test_optional_components_and_reconfiguration(self):
        backend = {EXECUTORCH / ('backends/arm/runtime/' + name) for name in (
            'EthosUBackend.cpp', 'EthosUBackend_Cortex_M.cpp',
            'EthosUBackend_IoMemcpy.cpp', 'VelaBinStream.cpp',
        )}
        tensor = {EXECUTORCH / ('extension/tensor/' + name) for name in ('tensor_ptr.cpp', 'tensor_ptr_maker.cpp')}
        config, build, sources = self.configure('components', extra='''
CONFIG_ETHOS_U=y
CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y
CONFIG_CMSIS_EXECUTORCH_EXTENSION_TENSOR=y
CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION=n
CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE=n
''')
        self.assertTrue((backend | tensor).issubset(sources))
        self.assertEqual(config.syms['CMSIS_EXECUTORCH_RUNTIME'].str_value, 'y')
        self.assertFalse(any(p.name == 'register_kernels.cpp' for p in sources))
        self.assertFalse(any('/kernels/portable/cpu/util/' in p.as_posix() for p in sources))
        definitions = (build / 'definitions.txt').read_text().split(';')
        self.assertIn('EXECUTORCH_BUILD_ARM_BAREMETAL=1', definitions)
        self.assertIn('ET_USE_ETHOS_U_BACKEND=1', definitions)
        _, build, sources = self.configure('components')
        self.assertFalse((backend | tensor) & sources)
        self.assertNotIn('ET_USE_ETHOS_U_BACKEND=1', (build / 'definitions.txt').read_text())

    def test_backend_requires_driver_and_cortex_m(self):
        for name, extra in [('no_driver', ''), ('not_cortex_m', 'CONFIG_CPU_CORTEX_M=n\nCONFIG_ETHOS_U=y\n')]:
            config, build, sources = self.configure(name, extra=extra + 'CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y\n')
            self.assertEqual(config.syms['CMSIS_EXECUTORCH_BACKEND_ETHOS_U'].str_value, 'n')
            self.assertFalse(any('/backends/arm/' in p.as_posix() for p in sources))
            self.assertNotIn('ET_USE_ETHOS_U_BACKEND=1', (build / 'definitions.txt').read_text())

    def test_runtime_and_utilities_can_be_controlled(self):
        no_defaults = 'CONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE=n\nCONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION=n\n'
        config, _, sources = self.configure('no_components', extra=no_defaults + 'CONFIG_CMSIS_EXECUTORCH_RUNTIME=n\n')
        self.assertFalse(sources)
        self.assertEqual(config.syms['CMSIS_EXECUTORCH_RUNTIME'].str_value, 'n')
        config, _, sources = self.configure('runtime_only', extra=no_defaults)
        self.assertIn(EXECUTORCH / 'runtime/platform/default/zephyr.cpp', sources)
        self.assertFalse(any('/kernels/portable/cpu/' in p.as_posix() for p in sources))
        config, _, sources = self.configure('operator_no_registration', ['PORTABLE_ADD'], extra=no_defaults)
        self.assertEqual(config.syms['CMSIS_EXECUTORCH_KERNEL_UTILS'].str_value, 'y')
        self.assertTrue(any('/kernels/portable/cpu/util/' in p.as_posix() for p in sources))
        self.assertFalse(any(p.name == 'register_kernels.cpp' for p in sources))

    def test_runtime_macro_defaults_and_disabled_values(self):
        _, build, _ = self.configure('runtime_flags')
        definitions = set((build / 'app_definitions.txt').read_text().split(';'))
        expected = {
            'ET_LOG_ENABLED=1', 'ET_MIN_LOG_LEVEL=Info',
            'ET_ENABLE_PROGRAM_VERIFICATION=1', 'ET_ENABLE_ENUM_STRINGS=1',
            'ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=1', 'ET_MAX_INSTRUCTIONS=10000000',
        }
        self.assertTrue(expected.issubset(definitions), definitions)
        presence_flags = {
            'ET_EVENT_TRACER_ENABLED', 'ET_OP_REGISTRY_DEBUG',
            'PROFILING_ENABLED', 'ET_ARM_ETHOSU_PER_DELEGATE_PROFILING',
        }
        self.assertFalse(any(d.split('=')[0] in presence_flags for d in definitions))
        _, build, _ = self.configure('runtime_flags', extra='''
CONFIG_CMSIS_EXECUTORCH_LOG_ENABLED=n
CONFIG_CMSIS_EXECUTORCH_PROGRAM_VERIFICATION=n
CONFIG_CMSIS_EXECUTORCH_ENUM_STRINGS=n
CONFIG_CMSIS_EXECUTORCH_DEPRECATED_CONSTANT_BUFFER=n
''')
        definitions = set((build / 'app_definitions.txt').read_text().split(';'))
        for macro in ('ET_LOG_ENABLED', 'ET_ENABLE_PROGRAM_VERIFICATION',
                      'ET_ENABLE_ENUM_STRINGS', 'ET_ENABLE_DEPRECATED_CONSTANT_BUFFER'):
            self.assertIn(macro + '=0', definitions)
            self.assertNotIn(macro + '=1', definitions)

    def test_runtime_instrumentation_and_reconfiguration(self):
        config, build, _ = self.configure('instrumentation', extra='''
CONFIG_ETHOS_U=y
CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y
CONFIG_CMSIS_EXECUTORCH_EVENT_TRACER=y
CONFIG_CMSIS_EXECUTORCH_OP_REGISTRY_DEBUG=y
CONFIG_CMSIS_EXECUTORCH_PROFILING=y
CONFIG_CMSIS_EXECUTORCH_ETHOS_U_PER_DELEGATE_PROFILING=y
CONFIG_CMSIS_EXECUTORCH_MAX_INSTRUCTIONS=12345
CONFIG_CMSIS_EXECUTORCH_MAX_PROFILE_EVENTS=17
CONFIG_CMSIS_EXECUTORCH_MAX_MEM_PROFILE_EVENTS=19
CONFIG_CMSIS_EXECUTORCH_MEM_PROFILE_MAX_ALLOCATORS=3
CONFIG_CMSIS_EXECUTORCH_MAX_PROFILE_BLOCKS=4
''')
        definitions = set((build / 'app_definitions.txt').read_text().split(';'))
        presence_flags = {
            'ET_EVENT_TRACER_ENABLED', 'ET_OP_REGISTRY_DEBUG',
            'PROFILING_ENABLED', 'ET_ARM_ETHOSU_PER_DELEGATE_PROFILING',
        }
        self.assertTrue(presence_flags.issubset(definitions))
        self.assertTrue({
            'ET_MAX_INSTRUCTIONS=12345', 'MAX_PROFILE_EVENTS=17',
            'MAX_MEM_PROFILE_EVENTS=19', 'MEM_PROFILE_MAX_ALLOCATORS=3',
            'MAX_PROFILE_BLOCKS=4',
        }.issubset(definitions))
        # The native options stay gated on EXECUTORCH even while our features
        # are enabled; their build must remain off to avoid duplicate sources.
        for symbol in ('EXECUTORCH', 'EXECUTORCH_ENABLE_EVENT_TRACER', 'EXECUTORCH_ENABLE_LOGGING'):
            self.assertEqual(config.syms[symbol].str_value, 'n')
        _, build, _ = self.configure('instrumentation')
        definitions = set((build / 'app_definitions.txt').read_text().split(';'))
        self.assertFalse(any(d.split('=')[0] in presence_flags for d in definitions))
        self.assertFalse(any(d.startswith('MAX_PROFILE_EVENTS=') for d in definitions))
        self.assertIn('ET_MAX_INSTRUCTIONS=10000000', definitions)

    def test_log_severities_and_profiler_defaults(self):
        for severity in ('Debug', 'Info', 'Error', 'Fatal'):
            _, build, _ = self.configure('log_level', extra=(
                f'CONFIG_CMSIS_EXECUTORCH_LOG_LEVEL_{severity.upper()}=y\n'
                'CONFIG_CMSIS_EXECUTORCH_PROFILING=y\n'
            ))
            definitions = set((build / 'app_definitions.txt').read_text().split(';'))
            self.assertEqual({d for d in definitions if d.startswith('ET_MIN_LOG_LEVEL=')},
                             {'ET_MIN_LOG_LEVEL=' + severity})
            self.assertTrue({
                'MAX_PROFILE_EVENTS=1024', 'MAX_MEM_PROFILE_EVENTS=1024',
                'MEM_PROFILE_MAX_ALLOCATORS=32', 'MAX_PROFILE_BLOCKS=2',
            }.issubset(definitions))

    @unittest.skipUnless(args.pack, 'Supply --pack for an independent PDSC coverage audit')
    def test_every_pack_component_and_source_is_covered(self):
        if args.pack.is_dir():
            manifest = ET.parse(args.pack / 'PyTorch.ExecuTorch.pdsc').getroot()
        else:
            with ZipFile(args.pack) as archive:
                manifest = ET.fromstring(archive.read('PyTorch.ExecuTorch.pdsc'))
        components = manifest.findall('./components/component')
        config_text = 'CONFIG_CMSIS_NN=y\nCONFIG_ETHOS_U=y\nCONFIG_CMSIS_EXECUTORCH_EMBEDDED_MODULE=n\n'
        symbols = []
        for component in components:
            guards = re.findall(r'(?m)^\s*#define\s+RTE_ML_EXECUTORCH_(\w+)\b', component.findtext('RTE_Components_h', ''))
            self.assertEqual(len(guards), 1, component.attrib)
            symbols.append('CMSIS_EXECUTORCH_' + guards[0])
        config_text += ''.join('CONFIG_' + symbol + '=y\n' for symbol in symbols)
        config, build, sources = self.configure('all_pack_components', extra=config_text)
        for symbol in symbols:
            self.assertIn(symbol, config.syms)
            self.assertEqual(config.syms[symbol].str_value, 'y', symbol)
        expected = set()
        for component in components:
            for entry in component.findall('./files/file'):
                if not entry.get('category', '').startswith('source'):
                    continue
                path = entry.get('name').removeprefix('src/')
                if path == 'runtime/platform/default/minimal.cpp':
                    path = 'runtime/platform/default/zephyr.cpp'
                if path == 'registration/RegisterAllKernels.cpp':
                    expected.add(build / 'module/generated/register_kernels.cpp')
                else:
                    expected.add(EXECUTORCH / path)
        self.assertEqual(sources, expected)
        print(f'\nPDSC audit: all {len(components)} components and {len(expected)} source files covered.')

    def test_artifact_mismatch_is_rejected(self):
        compatibility = self.module / 'zephyr/compatibility.cmake'
        original = compatibility.read_text()
        compatibility.write_text(re.sub(r'"[0-9a-f]{64}"', '"' + '0' * 64 + '"', original, count=1))
        try:
            message = self.configure('mismatch', expected_success=False)
            self.assertIn('differs from the schemas used', message)
        finally:
            compatibility.write_text(original)

    def test_upstream_build_is_mutually_exclusive(self):
        config, _, sources = self.configure('upstream', extra='CONFIG_EXECUTORCH=y\n')
        self.assertEqual(config.syms['CMSIS_EXECUTORCH'].str_value, 'n')
        self.assertFalse(sources)


unittest.main(argv=[sys.argv[0]], verbosity=2)

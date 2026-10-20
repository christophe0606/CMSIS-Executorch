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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zephyr-base', type=Path, required=True)
parser.add_argument('--executorch-dir', type=Path, required=True)
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
endfunction()
function(zephyr_library_named name)
  set_property(GLOBAL PROPERTY module_library "${name}")
endfunction()
function(zephyr_library_sources)
  set_property(GLOBAL APPEND PROPERTY module_sources ${ARGN})
endfunction()
function(zephyr_library_compile_definitions)
  set_property(GLOBAL APPEND PROPERTY module_definitions ${ARGN})
endfunction()
add_subdirectory("${TEST_MODULE}/zephyr" module)
get_property(sources GLOBAL PROPERTY module_sources)
get_property(definitions GLOBAL PROPERTY module_definitions)
get_property(includes GLOBAL PROPERTY module_includes)
get_property(library GLOBAL PROPERTY module_library)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sources.txt" CONTENT "${sources}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/definitions.txt" CONTENT "${definitions}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/includes.txt" CONTENT "${includes}")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/library.txt" CONTENT "${library}")
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
            if symbol.type in (kconfiglib.BOOL, kconfiglib.TRISTATE, kconfiglib.INT, kconfiglib.HEX):
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
        self.assertFalse(any('/kernels/portable/cpu/util/' in p.as_posix() for p in sources))
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

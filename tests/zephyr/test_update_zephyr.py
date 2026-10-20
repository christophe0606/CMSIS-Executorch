# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the release updater with small pack/check-out fixtures (stdlib only)."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'update_zephyr.py'
INPUTS = (
    'schema/program.fbs', 'schema/scalar_type.fbs',
    'kernels/portable/functions.yaml', 'kernels/portable/custom_ops.yaml',
    'kernels/quantized/quantized.yaml', 'backends/cortex_m/ops/operators.yaml',
)


def write(path, content, newline='\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline=newline)


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


class UpdateChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='update check ', dir=ROOT / 'tmp')
        self.work = Path(self.temporary.name).resolve()
        assert self.work.is_relative_to((ROOT / 'tmp').resolve())
        self.addCleanup(self.temporary.cleanup)
        self.pack = self.work / 'CMSIS pack'
        self.upstream = self.work / 'upstream/executorch'
        self.output = self.work / 'module/zephyr'
        write(self.upstream / 'version.txt', '9.2.3\n')
        write(self.upstream / 'third-party/flatbuffers/include/flatbuffers/base.h',
              '#define FLATBUFFERS_VERSION_MAJOR 24\n#define FLATBUFFERS_VERSION_MINOR 3\n#define FLATBUFFERS_VERSION_REVISION 25\n')
        for name in INPUTS:
            self.pair(name, '// fixture ' + name + '\n')
        for name in ('runtime/core/runtime.cpp', 'kernels/portable/cpu/util/helper.cpp',
                     'kernels/quantized/cpu/embeddingxb.cpp', 'kernels/portable/cpu/op_index_put.cpp',
                     'kernels/portable/cpu/index_helper.cpp', 'backends/cortex_m/ops/op_softmax.cpp'):
            self.pair(name, '// source ' + name + '\n')
        write(self.pack / 'src/runtime/platform/default/minimal.cpp', '// pack PAL\n')
        write(self.upstream / 'runtime/platform/default/zephyr.cpp', '// Zephyr PAL\n')
        for name in ('program_generated.h', 'scalar_type_generated.h'):
            write(self.pack / 'include/executorch/schema' / name,
                  '// generated schema\nstatic_assert(FLATBUFFERS_VERSION_MAJOR == 24 &&\n'
                  'FLATBUFFERS_VERSION_MINOR == 3 && FLATBUFFERS_VERSION_REVISION == 25);\n')
        self.manifest = self.pack / 'PyTorch.ExecuTorch.pdsc'
        write(self.manifest, '''<package><vendor>PyTorch</vendor><name>ExecuTorch</name>
<releases><release version="9.2.3"/><release version="1.0.0"/></releases>
<conditions><condition id="NN op"><require condition="CMSIS-NN"/></condition></conditions>
<components>
<component Cgroup="ExecuTorch" Csub="Runtime"><files>
  <file category="sourceCpp" name="src/runtime/core/runtime.cpp"/>
  <file category="sourceCpp" name="src/runtime/platform/default/minimal.cpp"/>
</files></component>
<component Cgroup="ExecuTorch" Csub="Kernel Utils"><files>
  <file category="sourceCpp" name="src/kernels/portable/cpu/util/helper.cpp"/>
  <file category="sourceCpp" name="src/kernels/quantized/cpu/embeddingxb.cpp"/>
</files></component>
<component Cgroup="ExecuTorch Operators" Csub="Index put">
  <RTE_Components_h>#define RTE_ML_EXECUTORCH_OP_PORTABLE_INDEX_PUT</RTE_Components_h><files>
  <file category="sourceCpp" name="src/kernels/portable/cpu/op_index_put.cpp"/>
  <file category="sourceCpp" name="src/kernels/portable/cpu/index_helper.cpp"/>
</files></component>
<component Cgroup="ExecuTorch Operators" Csub="Softmax" condition="NN op">
  <RTE_Components_h>#define RTE_ML_EXECUTORCH_OP_CORTEX_M_SOFTMAX</RTE_Components_h><files>
  <file category="sourceCpp" name="src/backends/cortex_m/ops/op_softmax.cpp"/>
</files></component>
</components></package>''')
        self.registration = self.pack / 'src/registration/RegisterAllKernels.cpp'
        write(self.registration, '''// Generated registration fixture
// Components use RTE_ML_EXECUTORCH_OP_PORTABLE_<NAME> guards.
#include "RTE_Components.h"
#ifdef RTE_ML_EXECUTORCH_OP_PORTABLE_INDEX_PUT
void declaration();
#endif
Kernel kernels[] = {
#ifdef RTE_ML_EXECUTORCH_OP_PORTABLE_INDEX_PUT
Kernel("index_put.out", [] {
#ifdef USE_ATEN_LIB
  aten();
#else
  portable();
#endif
}),
Kernel("index_put_", [] {}),
#endif // RTE_ML_EXECUTORCH_OP_PORTABLE_INDEX_PUT
#ifdef RTE_ML_EXECUTORCH_OP_CORTEX_M_SOFTMAX
Kernel("softmax.out", [] {}),
#endif // RTE_ML_EXECUTORCH_OP_CORTEX_M_SOFTMAX
Kernel(nullptr, nullptr)
};
''')

    def pair(self, name, content):
        write(self.pack / 'src' / name, content)
        # Windows checkouts must compare equal to LF pack inputs.
        write(self.upstream / name, content, newline='\r\n')

    def run_update(self, *extra, pack=None, expected=0):
        result = subprocess.run([
            sys.executable, str(SCRIPT), '--pack', str(pack or self.pack),
            '--executorch-dir', str(self.upstream), '--output-dir', str(self.output), *extra,
        ], cwd=self.work, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def assert_rejected_without_changes(self, expected_message):
        write(self.output / 'compatibility.cmake', 'existing release\n')
        before = snapshot(self.output)
        self.assertIn(expected_message, self.run_update(expected=2))
        self.assertEqual(snapshot(self.output), before)

    def test_directory_upgrade_from_other_working_directory(self):
        originals = snapshot(self.pack), snapshot(self.upstream)
        self.assertIn('9.2.3: 2 operator families, 3 registrations', self.run_update())
        artifacts = snapshot(self.output)
        self.assertEqual(len(artifacts), 7)
        compatibility = artifacts['compatibility.cmake'].decode()
        self.assertIn('EXPECTED_VERSION "9.2.3"', compatibility)
        digest = hashlib.sha256(('// fixture schema/program.fbs\n').encode()).hexdigest()
        self.assertIn(digest, compatibility)
        self.assertIn('PORTABLE_INDEX_PUT 2\n  kernels/portable/cpu/op_index_put.cpp\n  kernels/portable/cpu/index_helper.cpp)',
                      artifacts['operators.cmake'].decode())
        self.assertIn('select CMSIS_NN_SOFTMAX', artifacts['Kconfig.operators'].decode())
        self.assertIn('select CMSIS_NN_NNSUPPORT', artifacts['Kconfig.operators'].decode())
        registration = artifacts['include/cmsis_executorch/kernels.h'].decode()
        self.assertIn('#include <zephyr/autoconf.h>', registration)
        self.assertNotIn('RTE_Components.h', registration)
        self.assertNotIn('RTE_ML_EXECUTORCH_OP_', registration)
        sources = artifacts['sources.cmake'].decode()
        self.assertIn('runtime/platform/default/zephyr.cpp', sources)
        self.assertNotIn('minimal.cpp', sources)
        self.assertNotIn('embeddingxb.cpp', sources)
        self.assertTrue(all(b'\r\n' not in data for data in artifacts.values()))
        self.assertEqual((snapshot(self.pack), snapshot(self.upstream)), originals)

    def test_pack_archive_matches_directory(self):
        self.run_update()
        expected = snapshot(self.output)
        archive = self.work / 'PyTorch.ExecuTorch.9.2.3.pack'
        with ZipFile(archive, 'w') as bundle:
            for path in self.pack.rglob('*'):
                if path.is_file():
                    bundle.write(path, path.relative_to(self.pack).as_posix())
        self.output = self.work / 'archive output'
        self.run_update(pack=archive)
        self.assertEqual(snapshot(self.output), expected)

    def test_check_and_idempotence(self):
        self.assertIn('need refreshing', self.run_update('--check', expected=1))
        self.assertFalse(self.output.exists())
        self.run_update()
        stamps = {p: p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()}
        self.assertIn('up to date', self.run_update('--check'))
        self.assertIn('Updated 0 artifact', self.run_update())
        self.assertEqual({p: p.stat().st_mtime_ns for p in stamps}, stamps)
        write(self.output / 'compatibility.cmake', 'stale\n')
        self.assertIn('compatibility.cmake', self.run_update('--check', expected=1))
        self.assertEqual((self.output / 'compatibility.cmake').read_text(), 'stale\n')

    def test_version_mismatch(self):
        write(self.upstream / 'version.txt', '9.2.4\n')
        self.assert_rejected_without_changes('does not match upstream')

    def test_changed_inputs_rejected(self):
        for name in ('schema/program.fbs', 'kernels/portable/functions.yaml',
                     'runtime/core/runtime.cpp', 'kernels/portable/cpu/op_index_put.cpp'):
            with self.subTest(name=name):
                path = self.upstream / name
                original = path.read_bytes()
                try:
                    write(path, '// changed\n')
                    self.assert_rejected_without_changes('Pack and upstream checkout differ: ' + name)
                finally:
                    path.write_bytes(original)

    def test_missing_generated_header(self):
        (self.pack / 'include/executorch/schema/program_generated.h').unlink()
        self.assert_rejected_without_changes('Required file is missing')

    def test_flatbuffers_mismatch(self):
        write(self.upstream / 'third-party/flatbuffers/include/flatbuffers/base.h',
              '#define FLATBUFFERS_VERSION_MAJOR 25\n#define FLATBUFFERS_VERSION_MINOR 1\n#define FLATBUFFERS_VERSION_REVISION 0\n')
        self.assert_rejected_without_changes('different FlatBuffers version')

    def test_new_nn_operator_requires_mapping_review(self):
        write(self.manifest, self.manifest.read_text().replace('op_softmax.cpp', 'op_new.cpp'))
        self.assert_rejected_without_changes('CMSIS-NN function-group mapping')

    def test_registration_layout_change_requires_review(self):
        write(self.registration, self.registration.read_text().replace('#include "RTE_Components.h"', ''))
        self.assert_rejected_without_changes('missing RTE_Components.h')

    def test_conditional_kernel_registration_requires_review(self):
        write(self.registration, self.registration.read_text().replace(
            'Kernel("index_put_", [] {}),', '#ifdef NEW_VARIANT\nKernel("index_put_", [] {}),\n#endif'))
        self.assert_rejected_without_changes('Conditional kernel registration needs review')


if __name__ == '__main__':
    unittest.main(verbosity=2)

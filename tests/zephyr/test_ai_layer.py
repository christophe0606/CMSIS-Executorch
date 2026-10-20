# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check AI-layer backends and west discovery without exporting a real model.

Run with a Python environment containing PyYAML and west. Model compilation
and export are replaced with fixtures; no ML packages or target are used.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import create_ai_layer as generator
from ai_layer_backends import ExportedModel, cmsis, zephyr

spec = importlib.util.spec_from_file_location("west_ailayer_test", ROOT / "zephyr/ailayer.py")
west_ailayer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(west_ailayer)

MODEL = ExportedModel(b"fixture model\x00\x01", (
    "aten::_adaptive_avg_pool2d.out",
    "aten::add.out",
    "quantized_decomposed::quantize_per_tensor.out",
))


class AiLayerChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="AI layer checks ", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.app = Path(temporary.name).resolve()
        self.mlops = self.app / "solution.cbuild-mlops.yml"
        self.mlops.write_text("""cbuild-mlops:
  npu:
    type: Ethos-U85
  model:
    clayer: cmsis_layer/custom.clayer.yml
    input-shape: 1x16x16x3
""", encoding="utf-8")

    def generate(self, backend):
        with patch.object(generator, "compile_spec", return_value=object()), \
             patch.object(generator, "export_model", return_value=MODEL) as export:
            result = generator.create_ai_layer(self.mlops, backend=backend)
        return result, export

    def test_zephyr_layout_and_regeneration(self):
        with patch.object(cmsis, "pack_root", side_effect=AssertionError("Zephyr used a CMSIS pack")):
            entry, _ = self.generate("zephyr")
        layer = self.app / "ai_layer"
        self.assertEqual(entry, layer / "CMakeLists.txt")
        self.assertEqual({p.name for p in layer.iterdir()}, {
            "CMakeLists.txt", "model.pte", "model_pte.c", "model_pte.h",
            "components.conf", "operators.conf",
        })
        self.assertEqual((layer / "model.pte").read_bytes(), MODEL.pte)
        self.assertIn("0x00, 0x01", (layer / "model_pte.c").read_text())
        conf = (layer / "operators.conf").read_text()
        self.assertIn("CONFIG_CMSIS_EXECUTORCH_OP_PORTABLE_ADAPTIVE_AVG_POOL2D=y", conf)
        self.assertIn("CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_QUANTIZE=y", conf)
        self.assertNotIn("CONFIG_CMSIS_NN=y", (layer / "components.conf").read_text())
        with patch.object(generator, "compile_spec"), \
             patch.object(generator, "export_model", return_value=ExportedModel(b"new", ())):
            generator.create_ai_layer(self.mlops, backend="zephyr")
        self.assertNotIn("CONFIG_CMSIS_EXECUTORCH_OP_", (layer / "operators.conf").read_text())
        self.assertEqual((layer / "model.pte").read_bytes(), b"new")

    def test_cmsis_layout_uses_same_export(self):
        with patch.object(cmsis, "executorch_version", return_value="1.5.1"), \
             patch.object(cmsis, "components", return_value=(["Runtime"], ["Portable add"])):
            entry, export = self.generate("cmsis")
        self.assertEqual(entry, self.app / "cmsis_layer/custom.clayer.yml")
        self.assertEqual((entry.parent / "model.pte").read_bytes(), MODEL.pte)
        self.assertIn("Machine Learning:ExecuTorch Operators:Portable add", entry.read_text())
        self.assertFalse((entry.parent / "CMakeLists.txt").exists())
        self.assertEqual(export.call_args.args[1], {"input_shape": (1, 16, 16, 3)})

    def test_project_model_and_optional_model_settings(self):
        model_file = self.app / "model/model.py"
        model_file.parent.mkdir()
        model_file.write_text("# Model fixture; never imported by this test.\n")
        self.mlops.write_text("cbuild-mlops:\n  npu:\n    type: Ethos-U85\n")
        _, export = self.generate("zephyr")
        self.assertEqual(export.call_args.args[2], model_file)
        self.assertEqual(export.call_args.args[1], {})

    def test_unsupported_operator_does_not_change_existing_output(self):
        self.generate("zephyr")
        layer = self.app / "ai_layer"
        before = {p.name: p.read_bytes() for p in layer.iterdir()}
        with patch.object(generator, "compile_spec"), \
             patch.object(generator, "export_model", return_value=ExportedModel(b"bad", ("custom::missing.out",))):
            with self.assertRaisesRegex(ValueError, "unsupported operators: custom::missing.out"):
                generator.create_ai_layer(self.mlops, backend="zephyr")
        self.assertEqual(before, {p.name: p.read_bytes() for p in layer.iterdir()})

    def test_cortex_m_operator_enables_cmsis_nn(self):
        _, metadata = zephyr.render(self.mlops, {}, ExportedModel(b"", ("cortex_m::quantized_conv2d.out",)))
        self.assertIn("CONFIG_CMSIS_NN=y", metadata["components.conf"])

    def test_collects_all_methods_and_overloads(self):
        plans = [SimpleNamespace(operators=[SimpleNamespace(name="aten::add", overload="out")]),
                 SimpleNamespace(operators=[SimpleNamespace(name="custom::op", overload="")])]
        model = ExportedModel.from_program(SimpleNamespace(
            executorch_program=SimpleNamespace(execution_plan=plans), buffer=b"program",
        ))
        self.assertEqual(model, ExportedModel(b"program", ("aten::add.out", "custom::op")))

    def test_input_discovery(self):
        previous = Path.cwd()
        os.chdir(self.app)
        self.addCleanup(os.chdir, previous)
        self.assertEqual(west_ailayer.find_mlops_file(), self.mlops)
        other = self.app / "other.cbuild-mlops.yml"
        other.touch()
        with self.assertRaisesRegex(ValueError, "multiple MLOps files"):
            west_ailayer.find_mlops_file()
        self.assertEqual(west_ailayer.find_mlops_file(Path(self.mlops.name)), self.mlops)
        self.mlops.unlink()
        other.unlink()
        with self.assertRaisesRegex(ValueError, "no .* found"):
            west_ailayer.find_mlops_file()

    def test_west_routes_to_zephyr_backend(self):
        parser = argparse.ArgumentParser()
        command = west_ailayer.AiLayer()
        command.add_parser(parser.add_subparsers())
        args = parser.parse_args(["ailayer"])
        self.assertIsNone(args.mlops_file)
        with patch.object(west_ailayer, "find_mlops_file", return_value=self.mlops), \
             patch.object(generator, "create_ai_layer") as create:
            command.do_run(args, [])
        create.assert_called_once_with(self.mlops, backend="zephyr")

    @unittest.skipUnless(shutil.which("cmake"), "CMake is not installed")
    def test_cmake_configuration_scope_and_deferred_link(self):
        self.generate("zephyr")
        (self.app / "manual.conf").touch()
        (self.app / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(ai_layer_check NONE)
set(EXTRA_CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/manual.conf")
add_subdirectory(ai_layer)
# Observe the deferred link without requiring a host compiler or firmware.
function(target_link_libraries target scope library)
  if(NOT target STREQUAL "app" OR NOT scope STREQUAL "PRIVATE" OR NOT library STREQUAL "ai_layer")
    message(FATAL_ERROR "Unexpected AI-layer link")
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/linked.txt" "${target};${library}")
endfunction()
file(WRITE "${CMAKE_BINARY_DIR}/fragments.txt" "${EXTRA_CONF_FILE}")
get_target_property(model_sources ai_layer INTERFACE_SOURCES)
file(WRITE "${CMAKE_BINARY_DIR}/sources.txt" "${model_sources}")
''', encoding="utf-8")
        build = self.app / "build"
        result = subprocess.run(["cmake", "-S", str(self.app), "-B", str(build)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        fragments = (build / "fragments.txt").read_text().split(";")
        self.assertEqual([Path(p) for p in fragments], [
            self.app / "manual.conf", self.app / "ai_layer/components.conf", self.app / "ai_layer/operators.conf",
        ])
        self.assertEqual(Path((build / "sources.txt").read_text()), self.app / "ai_layer/model_pte.c")
        self.assertEqual((build / "linked.txt").read_text(), "app;ai_layer")


if __name__ == "__main__":
    unittest.main()

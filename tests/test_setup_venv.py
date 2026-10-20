# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check environment selection without installing packages or exporting models."""

from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup_venv as setup
import create_ai_layer as generator


class SetupChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="venv selection ", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        assert self.work.is_relative_to((ROOT / "tmp").resolve())
        self.app = self.work / "applications/my app"
        self.app.mkdir(parents=True)
        previous = Path.cwd()
        os.chdir(self.app)
        self.addCleanup(os.chdir, previous)

    def run_setup(self, *args, selected=None, usable=True):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [str(ROOT / "setup_venv.py"), *args]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(setup.shutil, "which", return_value="uv"))
            mocks = SimpleNamespace()
            for name in ("find_uv_venv", "venv_is_usable", "check_host_python", "check_venv_python", "pip", "smoke_test"):
                setattr(mocks, name, stack.enter_context(patch.object(setup, name)))
            mocks.find_uv_venv.return_value = selected
            mocks.venv_is_usable.return_value = usable
            mocks.builder = stack.enter_context(patch.object(setup.venv, "EnvBuilder"))
            mocks.run = stack.enter_context(patch.object(setup.subprocess, "run"))
            mocks.remove = stack.enter_context(patch.object(setup.shutil, "rmtree"))
            def remove_fixture(path):
                self.assertTrue(Path(path).resolve().is_relative_to(self.work))
                Path(path).rmdir()  # Only empty test directories; never a real venv.
            mocks.remove.side_effect = remove_fixture
            self.assertEqual(setup.main(), 0)
        return mocks

    def assert_requirements(self, mocks, target):
        for invocation in mocks.pip.call_args_list:
            self.assertEqual(invocation.args[0], setup.venv_python(target))
        requirements = [Path(c.args[c.args.index("-r") + 1]) for c in mocks.pip.call_args_list if "-r" in c.args]
        self.assertEqual(requirements, [ROOT / "requirements.txt", ROOT / "requirements-arm-tosa.txt"])
        mocks.smoke_test.assert_called_once_with(setup.venv_python(target))

    def test_pip_creates_in_current_directory_and_ignores_parent(self):
        parent = self.work / ".venv"
        parent.mkdir()
        mocks = self.run_setup(selected=parent)
        target = self.app / ".venv"
        mocks.find_uv_venv.assert_not_called()
        mocks.builder.return_value.create.assert_called_once_with(target)
        self.assert_requirements(mocks, target)

    def test_pip_checks_existing_local_environment(self):
        target = self.app / ".venv"
        target.mkdir()
        mocks = self.run_setup()
        mocks.venv_is_usable.assert_called_once_with(target)
        mocks.builder.assert_not_called()
        mocks.remove.assert_not_called()
        self.assert_requirements(mocks, target)

    def test_uv_reuses_parent_environment(self):
        target = self.work / ".venv"
        target.mkdir()
        mocks = self.run_setup("--uv", selected=target)
        mocks.find_uv_venv.assert_called_once()
        mocks.venv_is_usable.assert_called_once_with(target)
        mocks.builder.assert_not_called()
        mocks.run.assert_not_called()
        self.assertFalse((self.app / ".venv").exists())
        self.assert_requirements(mocks, target)

    def test_uv_creates_local_environment_if_none_found(self):
        for options, version, requested in ((('--uv',), '>=3.10,<3.14', None),
                                             (('--uv', '--python', '3.12'), '3.12', (3, 12))):
            with self.subTest(options=options):
                mocks = self.run_setup(*options)
                target = self.app / ".venv"
                self.assertEqual(mocks.run.call_args.args[0], ["uv", "venv", "--python", version, str(target)])
                mocks.check_venv_python.assert_called_once_with(setup.venv_python(target), requested)
                mocks.builder.assert_not_called()
                self.assert_requirements(mocks, target)

    def test_recreate_targets_the_selected_parent(self):
        target = self.work / ".venv"
        target.mkdir()
        mocks = self.run_setup("--uv", "--recreate", selected=target)
        mocks.remove.assert_called_once_with(target)
        self.assertEqual(mocks.run.call_args.args[0][-1], str(target))
        self.assert_requirements(mocks, target)

    def test_broken_pip_environment_is_recreated_in_working_directory(self):
        target = self.app / ".venv"
        target.mkdir()
        mocks = self.run_setup(usable=False)
        mocks.remove.assert_called_once_with(target)
        mocks.builder.return_value.create.assert_called_once_with(target)
        self.assert_requirements(mocks, target)

    def test_native_uv_discovery_accepts_only_virtual_environment(self):
        target = self.work / ".venv"
        target.mkdir()
        (target / "pyvenv.cfg").touch()
        selected_python = setup.venv_python(target)
        with patch.object(setup.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=str(selected_python) + "\n")) as run:
            self.assertEqual(setup.find_uv_venv("uv", {}), target)
        self.assertEqual(run.call_args.args[0], ["uv", "python", "find", "--no-project"])
        with patch.object(setup.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=str(self.work / "system/python.exe"))):
            self.assertIsNone(setup.find_uv_venv("uv", {}))

    def test_no_discoverable_python_allows_local_creation(self):
        with patch.object(setup.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")):
            self.assertIsNone(setup.find_uv_venv("uv", {}))

    def test_bootstrap_restores_original_active_environment(self):
        bootstrap = self.work / "bootstrap"
        active = self.work / "active environment"
        scripts = setup.venv_python(bootstrap).parent
        with patch.dict(os.environ, {
            "CMSIS_EXECUTORCH_SETUP_UV_BOOTSTRAP": "1",
            "CMSIS_EXECUTORCH_SETUP_VIRTUAL_ENV": str(active),
            "VIRTUAL_ENV": str(bootstrap),
            "PATH": os.pathsep.join([str(scripts), str(self.work / "tools")]),
        }, clear=True), patch.object(sys, "prefix", str(bootstrap)):
            env = setup.uv_environment()
        self.assertEqual(env["VIRTUAL_ENV"], str(active))
        self.assertEqual(env["PATH"], str(self.work / "tools"))
        self.assertNotIn("CMSIS_EXECUTORCH_SETUP_UV_BOOTSTRAP", env)

    def test_bootstrap_without_active_environment_allows_parent_discovery(self):
        with patch.dict(os.environ, {
            "CMSIS_EXECUTORCH_SETUP_UV_BOOTSTRAP": "1",
            "CMSIS_EXECUTORCH_SETUP_VIRTUAL_ENV": "",
            "VIRTUAL_ENV": "temporary-bootstrap",
        }, clear=True):
            self.assertNotIn("VIRTUAL_ENV", setup.uv_environment())

    def test_ai_layer_launcher_reuses_uv_parent(self):
        target = self.work / ".venv"
        python = setup.venv_python(target)
        python.parent.mkdir(parents=True)
        python.touch()
        with patch.object(generator.shutil, "which", return_value="uv"), \
             patch.object(generator, "find_uv_venv", return_value=target), \
             patch.object(generator.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                generator.run_in_venv()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(run.call_args.args[0][0], str(python))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Create the Python venv used by create_ai_layer.py to export the model. No Docker.
#
# Runs on Linux, macOS and Windows. The thin wrappers setup_venv.sh and
# setup_venv.bat just delegate here; everything OS-specific lives in this file.
"""Prepare the Python environment used to export models.

Requirements are always read from the directory containing this script.
Pip uses .venv in the working directory; --uv discovers an existing environment
using uv's normal search, including parent directories.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ExecuTorch 1.5 declares requires-python = ">=3.10,<3.15" in its pyproject.toml,
# but the tosa-tools 2026.5.0 it pins for the Arm backend (see
# requirements-arm-tosa.txt) only ships wheels up to CPython 3.13, so on 3.14
# pip reports "No matching distribution found for tosa-tools==2026.5.0".
# Check the version up front: without this, an unsupported interpreter fails
# much later with a resolver error that says nothing about the Python version.
MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 14)


def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a venv, on any host OS.

    Windows puts it in Scripts/python.exe, everyone else in bin/python. This is
    the one place that difference is encoded; run_in_venv() in
    create_ai_layer.py makes the same choice.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def uv_environment() -> dict[str, str]:
    """Restore the caller's environment when launched by an isolated uv wrapper."""
    env = dict(os.environ)
    bootstrap = env.pop("CMSIS_EXECUTORCH_SETUP_UV_BOOTSTRAP", None)
    original_venv = env.pop("CMSIS_EXECUTORCH_SETUP_VIRTUAL_ENV", "")
    if bootstrap:
        env.pop("VIRTUAL_ENV", None)
        if original_venv:
            env["VIRTUAL_ENV"] = original_venv
        # uv run also prepends its temporary interpreter to PATH.
        scripts = venv_python(Path(sys.prefix)).parent.resolve()
        env["PATH"] = os.pathsep.join(
            entry for entry in env.get("PATH", "").split(os.pathsep)
            if Path(entry).resolve() != scripts
        )
    return env


def find_uv_venv(uv: str, env: dict[str, str]) -> Path | None:
    """Ask uv to select Python; return its virtual environment, if any.

    Do not constrain the Python version during discovery: an existing but
    incompatible environment must be reported instead of silently skipped.
    --no-project follows the uv pip interface's environment selection rather
    than filtering interpreters through a project's Python requirement.
    """
    result = subprocess.run(
        [uv, "python", "find", "--no-project"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode:
        # No discoverable Python is needed to proceed: uv venv can download
        # the requested interpreter when creating the local environment.
        return None
    python = Path(result.stdout.strip())
    directory = python.parent.parent
    if (directory / "pyvenv.cfg").is_file():
        return directory
    return None


def check_host_python() -> None:
    if not (MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON_EXCLUSIVE):
        have = ".".join(str(n) for n in sys.version_info[:3])
        lo = ".".join(str(n) for n in MIN_PYTHON)
        hi = ".".join(str(n) for n in MAX_PYTHON_EXCLUSIVE)
        sys.exit(
            f"error: ExecuTorch needs Python >={lo},<{hi}; this is {have}\n"
            f"  ({sys.executable})\n"
            "Re-run with a supported interpreter, e.g.\n"
            "  PYTHON=python3.12 ./setup_venv.sh        (Linux/macOS)\n"
            "  py -3.12 setup_venv.py                   (Windows)"
        )


def venv_is_usable(venv_dir: Path) -> bool:
    """True if the venv exists and its interpreter still runs.

    A venv outlives the interpreter it was created from (a Python upgrade, a
    removed pyenv version): the directory is there but the symlinks and
    lib/pythonX.Y paths are stale.
    """
    python = venv_python(venv_dir)
    if not python.is_file():
        return False
    try:
        subprocess.run(
            [str(python), "-c", "import sys"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def python_version(value: str) -> tuple[int, ...]:
    """Accept a supported major.minor version, optionally with a patch version."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
        raise argparse.ArgumentTypeError("expected a Python version such as 3.12 or 3.12.10")
    version = tuple(int(n) for n in value.split("."))
    if not (MIN_PYTHON <= version[:2] < MAX_PYTHON_EXCLUSIVE):
        raise argparse.ArgumentTypeError("ExecuTorch needs Python >=3.10,<3.14")
    return version


def check_venv_python(python: Path, requested: tuple[int, ...] | None) -> None:
    """The venv's interpreter must be supported and, with --python, the one asked for."""
    result = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    )
    have = result.stdout.strip()
    version = tuple(int(n) for n in have.split("."))
    if not (MIN_PYTHON <= version[:2] < MAX_PYTHON_EXCLUSIVE):
        sys.exit(
            f"error: {python} uses Python {have}; ExecuTorch needs >=3.10,<3.14. "
            "Re-run with --recreate and a supported Python version."
        )
    if requested and version[: len(requested)] != requested:
        want = ".".join(map(str, requested))
        sys.exit(
            f"error: {python} uses Python {have}, but --python {want} was requested. "
            "Re-run with --recreate to change the environment's Python version."
        )


def pip(python: Path, *args: str, uv: str | None = None, env: dict[str, str] | None = None) -> None:
    if uv:
        cmd = [uv, "pip", *args, "--python", str(python)]
    else:
        cmd = [str(python), "-m", "pip", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def smoke_test(python: Path) -> None:
    """Import the module the export needs; a torch/executorch mismatch fails here."""
    try:
        subprocess.run(
            [str(python), "-c", "import executorch.backends.arm.quantizer.quantization_annotator"],
            check=True,
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "error: the executorch Arm quantizer does not import; the torch and "
            "executorch versions in .venv are probably mismatched.\n"
            f'Recreate it with: python "{HERE / "setup_venv.py"}" --recreate'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uv",
        action="store_true",
        help="use uv environment discovery (including parents), create if absent, and install with uv pip",
    )
    parser.add_argument(
        "--python",
        metavar="VERSION",
        type=python_version,
        help=(
            "Python version for uv, e.g. 3.12 or 3.12.10 (needs --uv; default: "
            "reuse a supported environment or let uv choose Python >=3.10,<3.14)"
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and rebuild the selected environment, including a parent environment with --uv",
    )
    args = parser.parse_args()

    if args.python and not args.uv:
        parser.error("--python requires --uv")
    uv = shutil.which("uv") if args.uv else None
    if args.uv and not uv:
        parser.error(
            "--uv requires uv on PATH; install it from "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    env = uv_environment() if uv else dict(os.environ)
    venv_dir = (find_uv_venv(uv, env) if uv else None) or Path.cwd() / ".venv"
    if not uv:
        check_host_python()

    if args.recreate and venv_dir.exists():
        print(f"Removing {venv_dir}")
        shutil.rmtree(venv_dir)

    if venv_dir.exists() and not venv_is_usable(venv_dir):
        print(f"{venv_dir} exists but its interpreter does not run; recreating.")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"Creating venv at {venv_dir}")
        if uv:
            # The wrapper may run this script in uv's temporary isolated
            # environment: request a version, not a path inside that environment.
            requested = ".".join(map(str, args.python)) if args.python else ">=3.10,<3.14"
            cmd = [uv, "venv", "--python", requested, str(venv_dir)]
            print(f"+ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=env)
        else:
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(venv_dir)

    python = venv_python(venv_dir)
    check_venv_python(python, args.python)
    if not uv:
        pip(python, "install", "--upgrade", "pip")

    # Pass 1: everything that resolves from PyPI.
    pip(python, "install", "-r", str(HERE / "requirements.txt"), uv=uv, env=env)

    # Pass 2: the TOSA serializer, without dependencies. See the header of
    # requirements-arm-tosa.txt for why --no-deps is load-bearing.
    # CMAKE_POLICY_VERSION_MINIMUM keeps an sdist fallback building under CMake 4.
    env = dict(env, CMAKE_POLICY_VERSION_MINIMUM="3.5")
    pip(
        python,
        "install",
        "--no-deps",
        "-r",
        str(HERE / "requirements-arm-tosa.txt"),
        uv=uv,
        env=env,
    )

    smoke_test(python)

    print()
    print(f"venv ready: {venv_dir}")
    print("Model-export interpreter:")
    print(f"  {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

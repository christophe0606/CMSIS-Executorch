#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Export a model from CMSIS MLOps information and generate an AI layer.

    python create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python create_ai_layer.py <solution>.cbuild-mlops.yml     # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The *.cbuild-mlops.yml is what CMSIS-Toolbox generates from the `mlops:` node
of the csolution. This script reads the NPU and Vela settings from it, exports
the PyTorch model in model/model.py for that NPU (quantize, delegate to
Ethos-U, compile with Vela) and writes the complete AI layer into the
directory of the clayer named under `model.clayer`. The extra keys of the
`model:` node (input-shape, calibration-samples) are model parameters that
CMSIS-Toolbox passes through to this script:

    ai_layer.clayer.yml   runtime, kernel utilities and registration, Ethos-U
                          backend and the operator components the exported
                          program actually uses
    model_pte.c / .h      the ExecuTorch program as a C array
    model.pte             the program itself, for inspection

The standalone CLI uses the working directory's .venv, or uv's discovered
environment when no local .venv exists (see setup_venv.py).
Python callers can instead import create_ai_layer(mlops_file); importing this
module does not export a model, load ML dependencies, or switch interpreters.
Library callers supply the required dependencies in their own environment.
The default CMSIS backend writes a .clayer.yml; backend="zephyr" writes
ai_layer/CMakeLists.txt, components.conf and operators.conf beside the input.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

from ai_layer_backends import ExportedModel, cmsis, zephyr
from setup_venv import find_uv_venv, uv_environment, venv_python

HERE = Path(__file__).resolve().parent
SYMBOL = "model_pte"


def run_in_venv() -> None:
    """Re-run under the local .venv, falling back to uv's discovered environment.

    Deciding by "does torch import" is not enough: a torch installed for the
    host interpreter would keep the export outside the environment with the
    pinned executorch. sys.prefix is the venv directory when running inside it
    (comparing interpreter paths does not work: venv symlinks resolve to the
    base interpreter).
    """
    venv = Path.cwd() / ".venv"
    if not venv.exists() and (uv := shutil.which("uv")):
        venv = find_uv_venv(uv, uv_environment()) or venv
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv_python(venv)
    if not python.is_file():
        sys.exit(
            f"{venv} does not exist.\n"
            f'Create it first: python "{HERE / "setup_venv.py"}"'
        )
    print(f"[ai_layer] running in {python}", flush=True)
    sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


# Vela options that are arguments of EthosUCompileSpec, and the ones the export
# sets itself: the configuration file comes from vela.ini of the cbuild-mlops.yml.
SPEC_OPTIONS = ("accelerator-config", "system-config", "memory-mode")
EXPORT_OPTIONS = ("config", "output-format", "output-dir")


def vela_options(options: str) -> tuple[dict[str, str], list[str]]:
    """Split vela.options of the cbuild-mlops.yml: the EthosUCompileSpec arguments, and every other option.

    The other options are the `misc:` of the csolution's mlops: node (for
    example `--optimise Size`); they reach Vela as extra flags, written as
    `--name=value`.
    """
    tokens = shlex.split(options)
    spec, extra = {}, []
    while tokens:
        token = tokens.pop(0)
        if not token.startswith("--"):
            raise ValueError(f"vela options: unexpected '{token}' in '{options}'")
        name, has_value, value = token[2:].partition("=")
        if not has_value and tokens and not tokens[0].startswith("-"):
            value = tokens.pop(0)
        if name in EXPORT_OPTIONS:
            raise ValueError(
                f"vela options: --{name} is set by the export; "
                "the configuration file is vela: ini: of the mlops: node"
            )
        if name in SPEC_OPTIONS:
            if name in spec:
                raise ValueError(f"vela options: --{name} is given twice in '{options}'")
            spec[name] = value
        else:
            extra.append(f"--{name}={value}" if value else f"--{name}")
    return spec, extra


def compile_spec(mlops: dict, mlops_dir: Path):
    """EthosUCompileSpec from the npu: and vela: nodes of the cbuild-mlops.yml."""
    npu = mlops.get("npu")
    if not npu:
        raise ValueError("the solution's mlops: node names no NPU; this template needs an Ethos-U")

    from executorch.backends.arm.ethosu import EthosUCompileSpec
    vela = mlops.get("vela", {})
    spec, extra = vela_options(vela.get("options", ""))

    target = spec.get("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": spec.get("system-config"),
        "memory_mode": spec.get("memory-mode"),
        "extra_flags": extra,
    }
    if vela.get("ini"):
        # ExecuTorch stores the path in the compile spec, and the spec ends up
        # in the .pte. A path relative to the working directory keeps the
        # program identical between checkouts; Vela resolves it from there.
        kwargs["config_ini"] = os.path.relpath(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def model_params(mlops: dict) -> dict:
    """The extra keys of the model: node, as keyword arguments for model/model.py."""
    params = {}
    for key, value in mlops.get("model", {}).items():
        if key == "input-shape":
            params["input_shape"] = tuple(int(d) for d in re.split(r"[x,]", str(value)))
        elif key == "calibration-samples":
            params["calibration_samples"] = int(value)
        elif key not in ("clayer", "name"):
            print(f"[ai_layer] warning: ignoring unknown model key {key!r}", file=sys.stderr)
    if params:
        print(f"[ai_layer] model parameters: {params}")
    return params


def export_model(spec, params: dict, model_file: Path) -> ExportedModel:
    """Quantize the model, delegate it to Ethos-U, and collect its CPU operators."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    module_name = "_ai_layer_model"
    module_spec = importlib.util.spec_from_file_location(module_name, model_file)
    model_module = importlib.util.module_from_spec(module_spec)
    # Register the module for Python decorators and Torch's source inspection.
    sys.modules[module_name] = model_module
    module_spec.loader.exec_module(model_module)
    model = model_module.get_model(**{k: v for k, v in params.items() if k == "input_shape"})
    samples = model_module.get_calibration_inputs(**params)
    example = (samples[0],)
    graph = torch.export.export(model, example).module()

    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only a float<->int8 boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)  # calibrate
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return ExportedModel.from_program(program)


def c_array(pte: bytes) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f"__attribute__((aligned(16))) const unsigned char {SYMBOL}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned long {SYMBOL}_size = sizeof({SYMBOL});\n"
    )


HEADER = f"""// Generated by create_ai_layer.py -- do not edit.
#pragma once
#ifdef __cplusplus
extern "C" {{
#endif
extern const unsigned char {SYMBOL}[];
extern const unsigned long {SYMBOL}_size;
#ifdef __cplusplus
}}
#endif
"""


def create_ai_layer(mlops_file: str | Path, *, backend: str = "cmsis") -> Path:
    """Export the model and write a CMSIS or Zephyr layer; return its entry point.

    Relative paths are resolved from the caller's working directory. This
    function uses the current Python environment and raises exceptions on
    failure; only the standalone CLI selects a local or uv-discovered environment.
    Both backends use the same MLOps input and model export. Zephyr writes
    ai_layer/ next to the input, independently of CMSIS's model.clayer field.
    """
    generators = {"cmsis": cmsis, "zephyr": zephyr}
    if backend not in generators:
        raise ValueError(f"unknown AI-layer backend: {backend}")
    mlops_file = Path(mlops_file).resolve()
    if not mlops_file.name.endswith(".cbuild-mlops.yml"):
        raise ValueError("expected a <solution>.cbuild-mlops.yml file")

    import yaml

    try:
        data = yaml.safe_load(mlops_file.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"{mlops_file}: invalid YAML: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("cbuild-mlops"), dict):
        raise ValueError(f"{mlops_file}: expected a cbuild-mlops mapping")
    mlops = data["cbuild-mlops"]
    if not isinstance(mlops.setdefault("model", {}), dict):
        raise ValueError(f"{mlops_file}: model must be a mapping")

    model_file = mlops_file.parent / "model/model.py"
    if not model_file.is_file():
        model_file = HERE / "model/model.py"
    spec = compile_spec(mlops, mlops_file.parent)
    params = model_params(mlops)
    # Preserve imports of helper modules next to model.py for the whole export,
    # then restore the library caller's search path even when export fails.
    sys.path.insert(0, str(model_file.parent))
    try:
        model = export_model(spec, params, model_file)
    finally:
        sys.path.pop(0)
    layer_file, metadata = generators[backend].render(mlops_file, mlops, model)
    layer_dir = layer_file.parent

    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "model.pte").write_bytes(model.pte)
    (layer_dir / f"{SYMBOL}.c").write_text(c_array(model.pte), encoding="utf-8", newline="\n")
    (layer_dir / f"{SYMBOL}.h").write_text(HEADER, encoding="utf-8", newline="\n")
    for name, contents in metadata.items():
        (layer_dir / name).write_text(contents, encoding="utf-8", newline="\n")

    print(f"[ai_layer] {len(model.pte)} byte program, operators: {list(model.operators)}")
    print(f"[ai_layer] wrote {layer_file}")
    return layer_file


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")
    run_in_venv()
    try:
        create_ai_layer(sys.argv[1])
    except (OSError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()

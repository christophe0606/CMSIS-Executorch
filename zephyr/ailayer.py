# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""West entry point for the shared AI-layer generator."""

from pathlib import Path
import sys

from west.commands import WestCommand


def find_mlops_file(path: Path | None = None) -> Path:
    """Use an explicit input or the only MLOps file in the working directory."""
    if path is not None:
        result = Path(path).resolve()
        if not result.name.endswith(".cbuild-mlops.yml") or not result.is_file():
            raise ValueError(f"MLOps input is not a *.cbuild-mlops.yml file: {result}")
        return result
    matches = sorted(p.resolve() for p in Path.cwd().glob("*.cbuild-mlops.yml") if p.is_file())
    if not matches:
        raise ValueError(f"no *.cbuild-mlops.yml found in {Path.cwd()}; supply the input path")
    if len(matches) > 1:
        raise ValueError(
            "multiple MLOps files found; supply one explicitly: "
            + ", ".join(p.name for p in matches)
        )
    return matches[0]


class AiLayer(WestCommand):
    def __init__(self):
        super().__init__(
            "ailayer",
            "create a Zephyr AI layer from CMSIS MLOps information",
            "Export a model and generate ai_layer/ with CMake integration and "
            "component/operator Kconfig fragments beside the MLOps input. "
            "Without an input path, use the only *.cbuild-mlops.yml in the current directory.",
        )

    def do_add_parser(self, parser_adder):
        parser = parser_adder.add_parser(
            self.name, help=self.help, description=self.description
        )
        parser.add_argument(
            "mlops_file", nargs="?", type=Path, metavar="<solution>.cbuild-mlops.yml",
            help="CMSIS-Toolbox MLOps description (default: discover in the current directory)",
        )
        return parser

    def do_run(self, args, unknown):
        try:
            mlops_file = find_mlops_file(args.mlops_file)
        except (OSError, ValueError) as error:
            self.die(str(error))

        # West loads this file by path, without adding the project root to
        # sys.path. Import the shared implementation from this checkout.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            from create_ai_layer import create_ai_layer
        finally:
            sys.path.pop(0)

        try:
            create_ai_layer(mlops_file, backend="zephyr")
        except (OSError, ValueError) as error:
            self.die(str(error))

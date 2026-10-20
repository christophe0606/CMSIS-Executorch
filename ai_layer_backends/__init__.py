# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared export result and operator selection for AI-layer generators."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ExportedModel:
    pte: bytes
    # Fully qualified names, including overloads, from all execution plans.
    operators: tuple[str, ...]

    @classmethod
    def from_program(cls, program):
        """Collect CPU operators from every method before serializing the PTE."""
        operators = tuple(sorted({
            op.name + (f".{op.overload}" if op.overload else "")
            for plan in program.executorch_program.execution_plan
            for op in plan.operators
        }))
        return cls(bytes(program.buffer), operators)


def selected_symbols(operators: tuple[str, ...], registration: str, prefix: str) -> list[str]:
    """Map program operators to the component guards in generated registration."""
    catalog = {}
    for block in re.finditer(
        rf"(?m)^\s*#ifdef ({prefix}[A-Z0-9_]+)\s*\n(.*?)^\s*#endif\b",
        registration, re.DOTALL,
    ):
        symbol, contents = block.groups()
        for operator in re.findall(r'\bKernel\s*\(\s*"([^"]+)"', contents):
            if operator in catalog and catalog[operator] != symbol:
                raise ValueError(f"ambiguous component registration for {operator}")
            catalog[operator] = symbol
    if not catalog:
        raise ValueError("no operator registrations found in the component catalog")
    missing = sorted(set(operators) - catalog.keys())
    if missing:
        raise ValueError(f"unsupported operators: {', '.join(missing)}")
    return sorted({catalog[operator].removeprefix(prefix) for operator in operators})

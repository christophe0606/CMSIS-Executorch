#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Refresh Zephyr artifacts from a matching CMSIS pack and ExecuTorch checkout.

This is a maintainer command, separate from firmware builds. It uses only the
Python standard library and never runs flatc, flatcc, or upstream codegen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


HERE = Path(__file__).resolve().parent
INPUTS = (
    'schema/program.fbs',
    'schema/scalar_type.fbs',
    'kernels/portable/functions.yaml',
    'kernels/portable/custom_ops.yaml',
    'kernels/quantized/quantized.yaml',
    'backends/cortex_m/ops/operators.yaml',
)
SCHEMA_HEADERS = ('program_generated.h', 'scalar_type_generated.h')
GROUPS = {
    'PORTABLE': 'Portable operators',
    'QUANTIZED': 'Quantized operators',
    'CORTEX_M': 'Cortex-M operators',
}
NN_LAYERS = {
    'maximum': 'BASICMATH',
    'minimum': 'BASICMATH',
    'pad': 'PAD',
    'quantized_activation': 'NNSUPPORT',
    'quantized_add': 'BASICMATH',
    'quantized_avg_pool2d': 'POOLING',
    'quantized_batch_matmul': 'FULLYCONNECTED',
    'quantized_conv2d': 'CONVOLUTION',
    'quantized_depthwise_conv2d': 'CONVOLUTION',
    'quantized_div': 'NNSUPPORT',
    'quantized_linear': 'FULLYCONNECTED',
    'quantized_max_pool2d': 'POOLING',
    'quantized_mul': 'BASICMATH',
    'quantized_transpose_conv2d': 'CONVOLUTION',
    'softmax': 'SOFTMAX',
    'transpose': 'TRANSPOSE',
}
GUARD = r'RTE_ML_EXECUTORCH_OP_[A-Z0-9_]+'
# Non-operator components have integration-specific dependencies/adapters.
# Fail on additions so a release refresh cannot silently omit pack components.
COMPONENTS = {
    'Runtime': ('RUNTIME', '\tdefault y\n'),
    'Kernel Utils': ('KERNEL_UTILS', '\tselect CMSIS_EXECUTORCH_RUNTIME\n'),
    'Extension Tensor': ('EXTENSION_TENSOR', '\tselect CMSIS_EXECUTORCH_RUNTIME\n'),
    'Kernel Registration': ('KERNEL_REGISTRATION', '\tdefault y\n\tselect CMSIS_EXECUTORCH_KERNEL_UTILS\n'),
    'Backend EthosU': ('BACKEND_ETHOS_U', '\tdepends on CPU_CORTEX_M && ETHOS_U\n\tselect CMSIS_EXECUTORCH_RUNTIME\n'),
}


class UpdateError(Exception):
    """An incomplete or incompatible input needs maintainer attention."""


def relative_path(name: str) -> str:
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in name or not path.parts:
        raise UpdateError(f'Invalid relative pack path: {name!r}')
    return path.as_posix()


def read_text(path: Path) -> str:
    if not path.is_file():
        raise UpdateError(f'Required file is missing: {path}')
    return path.read_bytes().replace(b'\r\n', b'\n').decode('utf-8')


class Pack:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.archive = None

    def __enter__(self):
        if self.path.is_dir():
            return self
        if not self.path.is_file():
            raise UpdateError(f'CMSIS pack does not exist: {self.path}')
        self.archive = ZipFile(self.path)
        return self

    def __exit__(self, *_):
        if self.archive is not None:
            self.archive.close()

    def read(self, name: str) -> str:
        name = relative_path(name)
        if self.archive is None:
            return read_text(self.path / name)
        try:
            data = self.archive.read(name)
        except KeyError:
            raise UpdateError(f'Required file is missing from {self.path}: {name}') from None
        return data.replace(b'\r\n', b'\n').decode('utf-8')


@dataclass(frozen=True)
class Operator:
    symbol: str
    label: str
    group: str
    sources: tuple[str, ...]
    kernel_count: int
    nn_layers: tuple[str, ...]


def source_paths(component: ET.Element) -> tuple[str, ...]:
    sources = []
    for entry in component.findall('./files/file'):
        if not entry.get('category', '').startswith('source'):
            continue
        if entry.get('condition'):
            raise UpdateError(f'Conditional source needs review: {entry.attrib}')
        name = relative_path(entry.get('name', ''))
        if not name.startswith('src/'):
            raise UpdateError(f'Expected a source under the pack src/ directory: {name}')
        sources.append(name.removeprefix('src/'))
    if not sources:
        raise UpdateError(f'No sources in pack component {component.get("Csub")!r}')
    return tuple(sources)


def matching_input(pack: Pack, upstream: Path, name: str) -> str:
    original = read_text(upstream / name)
    if original != pack.read('src/' + name):
        raise UpdateError(f'Pack and upstream checkout differ: {name}')
    return original


def registration_counts(registration: str) -> dict[str, int]:
    counts = defaultdict(int)
    active_guard = None
    depth = 0
    for line in registration.splitlines():
        opening = re.match(r'\s*#\s*(if|ifdef|ifndef)\b', line)
        closing = re.match(r'\s*#\s*endif\b', line)
        if active_guard:
            if opening:
                depth += 1
            elif closing:
                depth -= 1
                if depth == 0:
                    active_guard = None
            if re.match(r'\s*Kernel\s*\(', line):
                if depth != 1:
                    raise UpdateError(f'Conditional kernel registration needs review: {active_guard}')
                counts[active_guard] += 1
        else:
            match = re.fullmatch(r'\s*#\s*ifdef\s+(' + GUARD + r')\s*', line)
            if match:
                active_guard = match.group(1)
                counts[active_guard] += 0
                depth = 1
    if active_guard:
        raise UpdateError(f'Unclosed operator guard in registration: {active_guard}')
    referenced_guards = set(re.findall(r'(?m)^\s*#.*?\b(' + GUARD + r')\b', registration))
    if referenced_guards != set(counts):
        raise UpdateError('Unsupported operator guard layout in registration')
    return dict(counts)


def requires_nn(condition_id: str, conditions: dict[str, ET.Element], seen=()) -> bool:
    if condition_id == 'CMSIS-NN':
        return True
    if condition_id in seen:
        raise UpdateError(f'Cycle in pack conditions: {condition_id}')
    condition = conditions.get(condition_id)
    if condition is None:
        raise UpdateError(f'Unknown pack condition: {condition_id}')
    return any(
        requires_nn(child.get('condition'), conditions, (*seen, condition_id))
        for child in condition if child.get('condition')
    )


def operators_from_pack(manifest: ET.Element, registration: str) -> list[Operator]:
    conditions = {condition.get('id'): condition for condition in manifest.findall('./conditions/condition')}
    counts = registration_counts(registration)
    operators = []
    guards = set()
    for component in manifest.findall('./components/component[@Cgroup="ExecuTorch Operators"]'):
        matches = re.findall(r'\b' + GUARD + r'\b', component.findtext('RTE_Components_h', ''))
        if len(matches) != 1:
            raise UpdateError(f'Expected one operator guard in component {component.get("Csub")!r}')
        guard = matches[0]
        if guard in guards:
            raise UpdateError(f'Duplicate operator guard: {guard}')
        guards.add(guard)
        symbol = guard.removeprefix('RTE_ML_EXECUTORCH_OP_')
        groups = [group for group in GROUPS if symbol.startswith(group + '_')]
        if len(groups) != 1 or not counts.get(guard):
            raise UpdateError(f'Unknown category or missing kernel registrations: {guard}')
        group = groups[0]
        sources = source_paths(component)
        layers = set()
        condition = component.get('condition')
        needs_nn = requires_nn(condition, conditions) if condition else False
        if needs_nn:
            if group != 'CORTEX_M':
                raise UpdateError(f'CMSIS-NN dependency needs review: {symbol}')
            for source in sources:
                name = Path(source).stem.removeprefix('op_')
                if name not in NN_LAYERS:
                    raise UpdateError(f'Add a Zephyr CMSIS-NN function-group mapping for {source}')
                layers.add(NN_LAYERS[name])
            layers.add('NNSUPPORT')
        operators.append(Operator(symbol, component.get('Csub', symbol), group, sources, counts[guard], tuple(sorted(layers))))
    if not operators:
        raise UpdateError('The pack has no ExecuTorch operator components')
    if set(counts) != guards:
        raise UpdateError(f'Registration guards without pack components: {sorted(set(counts) - guards)}')
    return operators


def flatbuffers_version(text: str, generated: bool) -> tuple[int, int, int]:
    numbers = []
    for part in ('MAJOR', 'MINOR', 'REVISION'):
        pattern = rf'FLATBUFFERS_VERSION_{part}\s*==\s*(\d+)' if generated else rf'(?m)^#define FLATBUFFERS_VERSION_{part}\s+(\d+)'
        match = re.search(pattern, text)
        if not match:
            raise UpdateError(f'Cannot read FlatBuffers {part} version from {"generated schema" if generated else "base.h"}')
        numbers.append(int(match.group(1)))
    return tuple(numbers)


def render_registration(registration: str, version: str) -> str:
    include = re.search(r'(?m)^#include ["<]RTE_Components\.h[">]\s*$', registration)
    if include is None:
        raise UpdateError('Unsupported registration layout: missing RTE_Components.h include')
    prefix = re.sub(r'/\*.*?\*/|//[^\n]*', '', registration[:include.start()], flags=re.S).strip()
    if prefix:
        raise UpdateError('Unsupported code before RTE_Components.h; review the registration adapter')
    body = registration[include.end():].lstrip('\n')
    body = body.replace('RTE_ML_EXECUTORCH_OP_', 'CONFIG_CMSIS_EXECUTORCH_OP_')
    return f'''/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Pre-generated kernel declarations and registrations from
 * PyTorch::ExecuTorch@{version} RegisterAllKernels.cpp, with Zephyr Kconfig guards.
 * Prepared by update_zephyr.py; run it separately from firmware builds.
 * Include in exactly one translation unit (created in the build directory).
 */
#pragma once

#include <zephyr/autoconf.h>

''' + body


def render_components(manifest: ET.Element, pack: Pack, upstream: Path, banner: str) -> dict[str, str]:
    components = {}
    for component in manifest.findall('./components/component'):
        if component.get('Cgroup') == 'ExecuTorch Operators':
            continue
        name = component.get('Csub')
        if component.get('Cgroup') != 'ExecuTorch' or name not in COMPONENTS:
            raise UpdateError(f'Add a Zephyr mapping for pack component: {component.attrib}')
        if name in components:
            raise UpdateError(f'Duplicate pack component: {name}')
        components[name] = component
    if set(components) != set(COMPONENTS):
        raise UpdateError(f'Missing pack components: {sorted(set(COMPONENTS) - set(components))}')

    kconfig = banner + '# One selection per non-operator pack component.\n'
    sources = banner + '# Component sources are compiled directly from the upstream module.\n'
    sources += 'set(CMSIS_EXECUTORCH_COMPONENTS ' + ' '.join(symbol for symbol, _ in COMPONENTS.values()) + ')\n'
    for name, (symbol, settings) in COMPONENTS.items():
        component = components[name]
        guards = re.findall(r'(?m)^\s*#define\s+(RTE_ML_EXECUTORCH_\w+)\b', component.findtext('RTE_Components_h', ''))
        if guards != ['RTE_ML_EXECUTORCH_' + symbol]:
            raise UpdateError(f'Unexpected component guard for {name}: {guards}')
        kconfig += f'\nconfig CMSIS_EXECUTORCH_{symbol}\n\tbool "{name}"\n' + settings + '\thelp\n'
        description = ' '.join(component.findtext('description', name).split())
        kconfig += f'\t  {description}\n'
        sources += f'\nset(CMSIS_EXECUTORCH_{symbol}_SOURCES\n'
        component_sources = source_paths(component)
        for source in component_sources:
            if source == 'registration/RegisterAllKernels.cpp':
                if symbol != 'KERNEL_REGISTRATION':
                    raise UpdateError(f'Unexpected registration source in {name}')
                source_path = '${generated_directory}/register_kernels.cpp'
            else:
                if source == 'runtime/platform/default/minimal.cpp':
                    source = 'runtime/platform/default/zephyr.cpp'
                    read_text(upstream / source)
                else:
                    matching_input(pack, upstream, source)
                source_path = '${CMSIS_EXECUTORCH_SOURCE_DIR}/' + source
                kconfig += f'\t  Compile {source}.\n'
            sources += f'  "{source_path}"\n'
        sources += ')\n'
        if symbol == 'RUNTIME':
            kconfig += '\t  Use the upstream Zephyr PAL in place of the pack minimal PAL.\n'
        elif symbol == 'KERNEL_REGISTRATION':
            if component_sources != ('registration/RegisterAllKernels.cpp',):
                raise UpdateError('Kernel Registration source layout needs review')
            kconfig += '\t  Use pre-generated Zephyr registration guards for selected operators.\n'
            kconfig += '\t  Disable when the application supplies its own registration.\n'
        elif symbol == 'BACKEND_ETHOS_U':
            kconfig += '\t  Enable the Zephyr Ethos-U driver and configure the NPU for your board.\n'
    return {'Kconfig.components': kconfig, 'sources.cmake': sources}


def prepare(pack: Pack, upstream: Path) -> tuple[dict[str, str], str, list[Operator]]:
    manifest = ET.fromstring(pack.read('PyTorch.ExecuTorch.pdsc'))
    if manifest.findtext('vendor') != 'PyTorch' or manifest.findtext('name') != 'ExecuTorch':
        raise UpdateError('Expected the PyTorch::ExecuTorch CMSIS pack')
    release = manifest.find('./releases/release')
    version = release.get('version', '') if release is not None else ''
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.+-]+)?', version):
        raise UpdateError(f'Invalid or missing pack release version: {version!r}')
    actual_version = read_text(upstream / 'version.txt').strip()
    if version != actual_version:
        raise UpdateError(f'Pack version {version} does not match upstream version {actual_version}')

    inputs = {name: matching_input(pack, upstream, name) for name in INPUTS}
    registration = pack.read('src/registration/RegisterAllKernels.cpp')
    operators = operators_from_pack(manifest, registration)
    banner = f'''# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Generated by update_zephyr.py from PyTorch::ExecuTorch@{version}.

'''
    outputs = {}
    compatibility = banner + f'set(CMSIS_EXECUTORCH_EXPECTED_VERSION "{version}")\n'
    for name, text in inputs.items():
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        compatibility += f'cmsis_executorch_check_input("{name}"\n  "{digest}")\n'
    outputs['compatibility.cmake'] = compatibility

    upstream_flatbuffers = flatbuffers_version(read_text(upstream / 'third-party/flatbuffers/include/flatbuffers/base.h'), False)
    for name in SCHEMA_HEADERS:
        header = pack.read('include/executorch/schema/' + name)
        if flatbuffers_version(header, True) != upstream_flatbuffers:
            raise UpdateError(f'{name} requires a different FlatBuffers version than the upstream checkout')
        outputs['include/executorch/schema/' + name] = header
    outputs['include/cmsis_executorch/kernels.h'] = render_registration(registration, version)

    catalog = banner
    kconfig = banner + '# One selection per pack operator component, including all its overloads.\n'
    for group, label in GROUPS.items():
        members = [operator for operator in operators if operator.group == group]
        if not members:
            continue
        kconfig += f'\nmenu "{label}"\n'
        for operator in members:
            for source in operator.sources:
                matching_input(pack, upstream, source)
            catalog += f'cmsis_executorch_operator({operator.symbol} {operator.kernel_count}\n'
            catalog += ''.join(f'  {source}\n' for source in operator.sources).rstrip('\n') + ')\n'
            label = operator.label.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
            kconfig += f'\nconfig CMSIS_EXECUTORCH_OP_{operator.symbol}\n\tbool "{label}"\n'
            kconfig += '\tselect CMSIS_EXECUTORCH_KERNEL_UTILS\n'
            if operator.group == 'CORTEX_M':
                kconfig += '\tdepends on CPU_CORTEX_M\n'
            if operator.nn_layers:
                kconfig += '\tdepends on CMSIS_NN\n'
                kconfig += ''.join(f'\tselect CMSIS_NN_{layer}\n' for layer in operator.nn_layers)
            kconfig += '\thelp\n'
            kconfig += ''.join(f'\t  Compile {source}.\n' for source in operator.sources)
            kconfig += f'\t  Register all {operator.kernel_count} kernel variant(s) in this component.\n'
        kconfig += '\nendmenu\n'
    outputs['operators.cmake'] = catalog
    outputs['Kconfig.operators'] = kconfig

    outputs.update(render_components(manifest, pack, upstream, banner))
    return outputs, version, operators


def update(outputs: dict[str, str], destination: Path, check: bool) -> list[str]:
    changed = [name for name, text in outputs.items() if not (destination / name).is_file() or read_text(destination / name) != text]
    if check:
        return changed
    for name in changed:
        target = destination / relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(outputs[name])
            temporary.replace(target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True, help='Installed/unpacked PyTorch::ExecuTorch directory or .pack archive')
    parser.add_argument('--executorch-dir', type=Path, required=True, help='Matching upstream ExecuTorch checkout, including its FlatBuffers headers')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'zephyr', help='Destination Zephyr integration directory (default: this repository/zephyr)')
    parser.add_argument('--check', action='store_true', help='Report stale artifacts without writing files; exit 1 when an update is needed')
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    upstream = args.executorch_dir.resolve()
    try:
        with Pack(args.pack) as pack:
            outputs, version, operators = prepare(pack, upstream)
        # Validate all inputs and render all outputs before changing any artifact.
        changed = update(outputs, destination, args.check)
    except (UpdateError, OSError, UnicodeError, ET.ParseError, BadZipFile) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(f'ExecuTorch {version}: {len(operators)} operator families, {sum(op.kernel_count for op in operators)} registrations')
    if args.check:
        if changed:
            print('Artifacts need refreshing:\n' + '\n'.join('  ' + name for name in changed))
            return 1
        print('Generated artifacts are up to date.')
    else:
        print(f'Updated {len(changed)} artifact(s) in {destination}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

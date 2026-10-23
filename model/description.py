from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING, TypeAlias
from torch.fx import GraphModule

if TYPE_CHECKING:
    from executorch.backends.arm.ethosu import EthosUCompileSpec

MetadataValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | torch.Tensor
    | torch.dtype
    | torch.layout
    | list["MetadataValue"]
    | dict[str, "MetadataValue"]
)

# Description of a model
@dataclass
class Model:
    # Identifier for the model (used as a unique name)
    name: str
    # None means the model is already quantized; otherwise provide calibration samples.
    # Function has get_calibration_inputs(
    # input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 2
    # ) -> list[torch.Tensor]
    calibration_inputs: Callable[[tuple[int, ...], int], list[torch.Tensor]] | None = None
    # If the model do not use the default Ethos-U quantization scheme 
    # For instance may use : get_symmetric_a16w8_quantization_config
    custom_quantization: Callable[[EthosUCompileSpec, GraphModule], GraphModule] | None = None
    model: nn.Module | None = None

# Description of metadata for a particular model or experiment
@dataclass
class Metadata:
    details: dict[str, MetadataValue]

# When a Pte file contains metadata or several models
@dataclass
class Pte:
    name: str
    models: list[Model]
    metadata: Metadata | None = None

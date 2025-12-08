"""
Configuration package for Neural Network Accelerator.

Provides YAML-based configuration loading with schema validation.
"""

from .schema import CONFIG_SCHEMA, validate_custom_constraints, is_power_of_two
from .loader import (
    AcceleratorConfig,
    PrecisionConfig,
    SigmoidConfig,
    ParallelismConfig,
    MemoryConfig,
    NetworkConfig,
    TrainingConfig,
    DenseLayerConfig,
    Conv2DLayerConfig,
    MaxPoolLayerConfig,
    AvgPoolLayerConfig,
    FlattenLayerConfig,
    ActivationLayerConfig,
    LayerConfig,
    load_config,
    load_config_from_string,
    find_config,
    auto_load_config,
    DEFAULT_CONFIG_NAME,
    CONFIG_PATTERN,
)

__all__ = [
    # Schema
    "CONFIG_SCHEMA",
    "validate_custom_constraints",
    "is_power_of_two",
    # Config classes
    "AcceleratorConfig",
    "PrecisionConfig",
    "SigmoidConfig",
    "ParallelismConfig",
    "MemoryConfig",
    "NetworkConfig",
    "TrainingConfig",
    # Layer configs
    "DenseLayerConfig",
    "Conv2DLayerConfig",
    "MaxPoolLayerConfig",
    "AvgPoolLayerConfig",
    "FlattenLayerConfig",
    "ActivationLayerConfig",
    "LayerConfig",
    # Functions
    "load_config",
    "load_config_from_string",
    "find_config",
    "auto_load_config",
    # Constants
    "DEFAULT_CONFIG_NAME",
    "CONFIG_PATTERN",
]

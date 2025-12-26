"""
Configuration Loader and Parser

Loads YAML configuration files, validates against schema,
and provides a structured configuration object.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
import glob as glob_module

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from .schema import CONFIG_SCHEMA, validate_custom_constraints, is_power_of_two


# Default config filename
DEFAULT_CONFIG_NAME = "default.nn_conf.yaml"
CONFIG_PATTERN = "*.nn_conf.yaml"
CONFIG_PATTERN_YML = "*.nn_conf.yml"


@dataclass
class PrecisionConfig:
    """Fixed-point precision configuration."""
    int_bits: int
    frac_bits: int
    
    @property
    def data_width(self) -> int:
        return self.int_bits + self.frac_bits


@dataclass
class SigmoidConfig:
    """Sigmoid LUT configuration."""
    lut_bits: int
    range_bits: int
    
    @property
    def lut_size(self) -> int:
        """Number of LUT entries (2^lut_bits)."""
        return 2 ** self.lut_bits
    
    @property
    def input_range(self) -> tuple[float, float]:
        """Input range as (min, max) tuple."""
        return (-2.0 ** self.range_bits, 2.0 ** self.range_bits)
    
    @property
    def index_high(self) -> int:
        """High bit for index extraction."""
        return self.range_bits
    
    @property
    def index_low(self) -> int:
        """Low bit for index extraction (can be negative for fractional)."""
        return self.range_bits + 1 - self.lut_bits
    
    @property
    def index_frac_bits(self) -> int:
        """Number of fractional bits needed for indexing."""
        return max(0, -self.index_low)
    
    def get_overflow_check_bits(self, int_bits: int) -> int:
        """Number of upper bits to check for overflow detection."""
        return int_bits - 1 - self.range_bits


@dataclass
class ParallelismConfig:
    """Hardware parallelism configuration."""
    conv_parallel_positions: int = 1
    batch_size: int = 1


@dataclass
class MemoryConfig:
    """Memory layout configuration."""
    bram_depth: int = 1024
    input_base_addr: int = 0
    weights_base_addr: int = 256
    output_base_addr: int = 61440  # 0xF000


@dataclass
class DenseLayerConfig:
    """Dense layer configuration."""
    type: str = "dense"
    neurons: int = 1
    activation: str = "sigmoid"


@dataclass
class Conv2DLayerConfig:
    """Conv2D layer configuration."""
    type: str = "conv2d"
    in_channels: Optional[int] = None
    out_channels: int = 1
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    activation: str = "relu"


@dataclass
class MaxPoolLayerConfig:
    """MaxPool layer configuration."""
    type: str = "maxpool"
    kernel_size: int = 2
    stride: Optional[int] = None
    
    def __post_init__(self):
        if self.stride is None:
            self.stride = self.kernel_size


@dataclass
class AvgPoolLayerConfig:
    """AvgPool layer configuration."""
    type: str = "avgpool"
    kernel_size: int = 2
    stride: Optional[int] = None
    
    def __post_init__(self):
        if self.stride is None:
            self.stride = self.kernel_size


@dataclass
class FlattenLayerConfig:
    """Flatten layer configuration."""
    type: str = "flatten"


@dataclass
class ActivationLayerConfig:
    """Standalone activation layer configuration."""
    type: str = "relu"  # or "sigmoid"


LayerConfig = Union[
    DenseLayerConfig,
    Conv2DLayerConfig,
    MaxPoolLayerConfig,
    AvgPoolLayerConfig,
    FlattenLayerConfig,
    ActivationLayerConfig
]


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    name: str
    layers: list[LayerConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.01
    epochs: int = 1000
    optimizer: str = "sgd"


@dataclass
class AcceleratorConfig:
    """Complete accelerator configuration."""
    precision: PrecisionConfig
    sigmoid: SigmoidConfig
    network: NetworkConfig
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: Optional[TrainingConfig] = None
    
    @property
    def data_width(self) -> int:
        return self.precision.data_width
    
    @property
    def int_bits(self) -> int:
        return self.precision.int_bits
    
    @property
    def frac_bits(self) -> int:
        return self.precision.frac_bits


def _parse_layer(layer_dict: dict) -> LayerConfig:
    """Parse a layer dictionary into the appropriate config dataclass."""
    layer_type = layer_dict.get("type")
    
    if layer_type == "dense":
        return DenseLayerConfig(
            neurons=layer_dict["neurons"],
            activation=layer_dict.get("activation", "sigmoid")
        )
    elif layer_type == "conv2d":
        return Conv2DLayerConfig(
            in_channels=layer_dict.get("in_channels"),
            out_channels=layer_dict["out_channels"],
            kernel_size=layer_dict["kernel_size"],
            stride=layer_dict.get("stride", 1),
            padding=layer_dict.get("padding", 0),
            activation=layer_dict.get("activation", "relu")
        )
    elif layer_type == "maxpool":
        return MaxPoolLayerConfig(
            kernel_size=layer_dict["kernel_size"],
            stride=layer_dict.get("stride")
        )
    elif layer_type == "avgpool":
        return AvgPoolLayerConfig(
            kernel_size=layer_dict["kernel_size"],
            stride=layer_dict.get("stride")
        )
    elif layer_type == "flatten":
        return FlattenLayerConfig()
    elif layer_type in ("relu", "sigmoid"):
        return ActivationLayerConfig(type=layer_type)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def _parse_config_dict(raw_config: dict) -> AcceleratorConfig:
    """Parse validated config dictionary into AcceleratorConfig."""
    precision_dict = raw_config["precision"]
    
    # Handle precision: either (int_bits + frac_bits) or total_bits
    if "total_bits" in precision_dict:
        total = precision_dict["total_bits"]
        int_bits = total // 2
        frac_bits = total // 2
    else:
        int_bits = precision_dict["int_bits"]
        frac_bits = precision_dict["frac_bits"]
    
    precision = PrecisionConfig(int_bits=int_bits, frac_bits=frac_bits)
    
    sigmoid = SigmoidConfig(
        lut_bits=raw_config["sigmoid"]["lut_bits"],
        range_bits=raw_config["sigmoid"]["range_bits"]
    )
    
    parallelism_dict = raw_config.get("parallelism", {})
    parallelism = ParallelismConfig(
        conv_parallel_positions=parallelism_dict.get("conv_parallel_positions", 1),
        batch_size=parallelism_dict.get("batch_size", 1)
    )
    
    memory_dict = raw_config.get("memory", {})
    memory = MemoryConfig(
        bram_depth=memory_dict.get("bram_depth", 1024),
        input_base_addr=memory_dict.get("input_base_addr", 0),
        weights_base_addr=memory_dict.get("weights_base_addr", 256),
        output_base_addr=memory_dict.get("output_base_addr", 61440)
    )
    
    # Parse network
    network_dict = raw_config["network"]
    layers = [_parse_layer(layer) for layer in network_dict["layers"]]
    network = NetworkConfig(
        name=network_dict["name"],
        layers=layers
    )
    
    # Parse training (optional)
    training = None
    if "training" in raw_config:
        training_dict = raw_config["training"]
        training = TrainingConfig(
            learning_rate=training_dict.get("learning_rate", 0.01),
            epochs=training_dict.get("epochs", 1000),
            optimizer=training_dict.get("optimizer", "sgd")
        )
    
    return AcceleratorConfig(
        precision=precision,
        sigmoid=sigmoid,
        parallelism=parallelism,
        memory=memory,
        network=network,
        training=training
    )


def _validate_config(raw_config: dict) -> None:
    """Validate configuration against schema and custom constraints."""
    if raw_config is None:
        raise ValueError("Empty configuration")
    
    # Validate against JSON schema
    if JSONSCHEMA_AVAILABLE:
        try:
            jsonschema.validate(raw_config, CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            raise ValueError(f"Configuration error at '{path}': {e.message}")
    else:
        print("WARNING: jsonschema not installed, skipping schema validation")
    
    # Validate custom constraints
    errors = validate_custom_constraints(raw_config)
    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


def load_config(config_path: Union[str, Path]) -> AcceleratorConfig:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        AcceleratorConfig object
        
    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    _validate_config(raw_config)
    return _parse_config_dict(raw_config)


def load_config_from_string(yaml_string: str) -> AcceleratorConfig:
    """
    Load and validate configuration from a YAML string.
    
    Useful for testing or programmatic configuration.
    """
    raw_config = yaml.safe_load(yaml_string)
    _validate_config(raw_config)
    return _parse_config_dict(raw_config)


def find_config(root_dir: Union[str, Path], explicit_config: Optional[str] = None) -> Path:
    """
    Find configuration file using lookup order:
    1. Explicit config path (if provided)
    2. Search root directory for *.nn_conf.yaml or *.nn_conf.yml
    3. Fall back to configs/default.nn_conf.yaml
    
    Args:
        root_dir: Project root directory
        explicit_config: Explicitly specified config path (optional)
        
    Returns:
        Path to configuration file
        
    Raises:
        FileNotFoundError: If no configuration file found
    """
    root_dir = Path(root_dir)
    
    # 1. Explicit config takes priority
    if explicit_config:
        explicit_path = Path(explicit_config)
        if not explicit_path.is_absolute():
            explicit_path = root_dir / explicit_path
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Explicit config not found: {explicit_config}")
    
    # 2. Search root directory for *.nn_conf.yaml or *.nn_conf.yml
    for pattern in [CONFIG_PATTERN, CONFIG_PATTERN_YML]:
        matches = list(root_dir.glob(pattern))
        if matches:
            if len(matches) > 1:
                print(f"WARNING: Multiple config files found, using: {matches[0].name}")
            return matches[0]
    
    # 3. Fall back to configs/default.nn_conf.yaml
    default_path = root_dir / "configs" / DEFAULT_CONFIG_NAME
    if default_path.exists():
        return default_path
    
    raise FileNotFoundError(
        f"No configuration file found. Searched:\n"
        f"  - {root_dir}/{CONFIG_PATTERN}\n"
        f"  - {root_dir}/{CONFIG_PATTERN_YML}\n"
        f"  - {default_path}"
    )


def auto_load_config(root_dir: Union[str, Path], explicit_config: Optional[str] = None) -> AcceleratorConfig:
    """
    Automatically find and load configuration.
    
    Convenience function that combines find_config and load_config.
    
    Args:
        root_dir: Project root directory
        explicit_config: Explicitly specified config path (optional)
        
    Returns:
        AcceleratorConfig object
    """
    config_path = find_config(root_dir, explicit_config)
    return load_config(config_path)

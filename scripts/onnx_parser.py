"""
ONNX Model Parser for Neural Network Accelerator

Parses ONNX model files to extract:
- Network topology (layer types, sizes)
- Weights and biases (converted to fixed-point)
- Activation functions per layer

Currently supports dense (fully-connected) networks with
Sigmoid and ReLU activations. Validates that all ops are
hardware-supported.
"""

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnx
    from onnx import numpy_helper

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ONNX ops we can map to hardware
SUPPORTED_ACTIVATIONS = {"Sigmoid", "Relu", "Tanh"}
SUPPORTED_DENSE_OPS = {"Gemm", "MatMul"}
SUPPORTED_CONV_OPS = {"Conv"}
SUPPORTED_POOL_OPS = {"MaxPool", "AveragePool", "GlobalAveragePool"}
SUPPORTED_RNN_OPS = {"RNN", "LSTM"}
SUPPORTED_OPS = (
    SUPPORTED_ACTIVATIONS | SUPPORTED_DENSE_OPS | SUPPORTED_CONV_OPS |
    SUPPORTED_POOL_OPS | SUPPORTED_RNN_OPS |
    {"Add", "Reshape", "Flatten", "Shape", "Gather", "Unsqueeze", "Concat",
     "Constant", "Squeeze", "Transpose", "BatchNormalization", "Dropout"}
)


@dataclass
class DenseLayerInfo:
    """Parsed info for a single dense layer."""

    layer_type: str = "dense"
    num_inputs: int = 0
    num_outputs: int = 0
    activation: str = "none"  # "sigmoid", "relu", "tanh", or "none"
    weights: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (num_inputs, num_outputs)
    biases: np.ndarray = field(default_factory=lambda: np.array([]))   # shape (num_outputs,)


@dataclass
class ConvLayerInfo:
    """Parsed info for a Conv2D layer."""

    layer_type: str = "conv2d"
    c_in: int = 1
    h_in: int = 0
    w_in: int = 0
    num_filters: int = 1
    kernel_h: int = 3
    kernel_w: int = 3
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    activation: str = "none"
    weights: np.ndarray = field(default_factory=lambda: np.array([]))  # (out_ch, in_ch, kH, kW)
    biases: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def h_out(self) -> int:
        return (self.h_in + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1

    @property
    def w_out(self) -> int:
        return (self.w_in + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1

    @property
    def num_outputs(self) -> int:
        return self.num_filters * self.h_out * self.w_out


@dataclass
class PoolLayerInfo:
    """Parsed info for a pooling layer."""

    layer_type: str = "maxpool"  # "maxpool" or "avgpool"
    c_in: int = 1
    h_in: int = 0
    w_in: int = 0
    kernel_h: int = 2
    kernel_w: int = 2
    stride_h: int = 2
    stride_w: int = 2
    activation: str = "none"

    @property
    def h_out(self) -> int:
        return (self.h_in - self.kernel_h) // self.stride_h + 1

    @property
    def w_out(self) -> int:
        return (self.w_in - self.kernel_w) // self.stride_w + 1

    @property
    def num_outputs(self) -> int:
        return self.c_in * self.h_out * self.w_out


@dataclass
class FlattenLayerInfo:
    """Parsed info for a flatten layer."""

    layer_type: str = "flatten"
    num_inputs: int = 0
    activation: str = "none"

    @property
    def num_outputs(self) -> int:
        return self.num_inputs


@dataclass
class RNNLayerInfo:
    """Parsed info for an RNN/LSTM layer."""

    layer_type: str = "rnn"  # "rnn" or "lstm"
    input_size: int = 0
    hidden_size: int = 0
    seq_len: int = 1
    activation: str = "none"
    weights_ih: np.ndarray = field(default_factory=lambda: np.array([]))
    weights_hh: np.ndarray = field(default_factory=lambda: np.array([]))
    biases: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_outputs(self) -> int:
        return self.hidden_size


# Union type for any layer info
LayerInfo = DenseLayerInfo | ConvLayerInfo | PoolLayerInfo | FlattenLayerInfo | RNNLayerInfo


@dataclass
class ParsedNetwork:
    """Complete parsed network ready for VHDL generation."""

    name: str
    num_inputs: int
    input_shape: tuple = ()  # (C, H, W) for images, (features,) for vectors
    layers: list[LayerInfo] = field(default_factory=list)

    @property
    def num_outputs(self) -> int:
        return self.layers[-1].num_outputs if self.layers else 0

    @property
    def layer_sizes(self) -> list[int]:
        return [layer.num_outputs for layer in self.layers]

    @property
    def activations(self) -> list[str]:
        return [layer.activation for layer in self.layers]


def _get_initializer(graph, name: str) -> Optional[np.ndarray]:
    """Get a weight tensor by name from graph initializers."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _get_attr(node, name: str, default=None):
    """Get an attribute value from an ONNX node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
    return default


def _check_unsupported_ops(graph) -> list[str]:
    """Return list of unsupported op types found in the graph."""
    unsupported = []
    for node in graph.node:
        if node.op_type not in SUPPORTED_OPS:
            unsupported.append(node.op_type)
    return unsupported


def _peek_activation(nodes, i, output_name) -> tuple[str, int]:
    """Check if the next node is an activation consuming output_name.
    Returns (activation_name, nodes_consumed)."""
    if i + 1 < len(nodes) and nodes[i + 1].op_type in SUPPORTED_ACTIVATIONS:
        if nodes[i + 1].input[0] == output_name:
            return nodes[i + 1].op_type.lower(), 1
    return "none", 0


def parse_onnx(model_path: str | Path) -> ParsedNetwork:
    """
    Parse an ONNX model file into a ParsedNetwork.

    Supports:
      - Dense: Gemm or MatMul+Add patterns
      - Conv2D: Conv op
      - Pooling: MaxPool, AveragePool, GlobalAveragePool
      - Flatten/Reshape
      - RNN/LSTM ops
      - Activations: Sigmoid, Relu, Tanh

    Args:
        model_path: Path to .onnx file

    Returns:
        ParsedNetwork with topology and weights

    Raises:
        ImportError: If onnx package not installed
        ValueError: If model contains unsupported ops
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "The 'onnx' package is required for ONNX model parsing. "
            "Install it with: pip install onnx"
        )

    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    graph = model.graph

    # Check for unsupported ops
    unsupported = _check_unsupported_ops(graph)
    if unsupported:
        raise ValueError(
            f"Model contains unsupported operations: {set(unsupported)}. "
            f"Supported: {SUPPORTED_OPS}"
        )

    # Determine input shape from graph inputs (skip initializer names)
    init_names = {init.name for init in graph.initializer}
    real_inputs = [inp for inp in graph.input if inp.name not in init_names]
    if len(real_inputs) != 1:
        raise ValueError(
            f"Expected exactly 1 network input, got {len(real_inputs)}: "
            f"{[i.name for i in real_inputs]}"
        )

    input_dims = real_inputs[0].type.tensor_type.shape.dim
    input_shape_raw = tuple(d.dim_value for d in input_dims)
    # Remove batch dimension
    if len(input_shape_raw) >= 2:
        input_shape = input_shape_raw[1:]
    else:
        input_shape = input_shape_raw

    num_inputs = 1
    for d in input_shape:
        num_inputs *= d if d > 0 else 1

    # Track spatial dimensions through the network
    # Start: (C, H, W) for images, or (features,) for vectors
    current_shape = input_shape

    network_name = graph.name or model_path.stem
    layers: list[LayerInfo] = []
    nodes = list(graph.node)

    i = 0
    while i < len(nodes):
        node = nodes[i]

        if node.op_type == "Gemm":
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            weights = _get_initializer(graph, weight_name)
            biases = _get_initializer(graph, bias_name) if bias_name else None

            if weights is None:
                raise ValueError(f"Could not find weights '{weight_name}'")

            transB = _get_attr(node, "transB", 0)
            if transB:
                weights = weights.T

            num_in = weights.shape[0]
            num_out = weights.shape[1]
            if biases is None:
                biases = np.zeros(num_out, dtype=np.float32)

            activation, skip = _peek_activation(nodes, i, node.output[0])
            i += skip

            layers.append(DenseLayerInfo(
                num_inputs=num_in, num_outputs=num_out,
                activation=activation, weights=weights, biases=biases,
            ))
            current_shape = (num_out,)

        elif node.op_type == "MatMul":
            weight_name = node.input[1]
            weights = _get_initializer(graph, weight_name)
            if weights is None:
                raise ValueError(f"Could not find weights '{weight_name}'")

            num_in = weights.shape[0]
            num_out = weights.shape[1]
            matmul_output = node.output[0]

            biases = np.zeros(num_out, dtype=np.float32)
            if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                add_node = nodes[i + 1]
                if add_node.input[0] == matmul_output:
                    bias_arr = _get_initializer(graph, add_node.input[1])
                    if bias_arr is not None:
                        biases = bias_arr
                    matmul_output = add_node.output[0]
                    i += 1

            activation, skip = _peek_activation(nodes, i, matmul_output)
            i += skip

            layers.append(DenseLayerInfo(
                num_inputs=num_in, num_outputs=num_out,
                activation=activation, weights=weights, biases=biases,
            ))
            current_shape = (num_out,)

        elif node.op_type == "Conv":
            weight_name = node.input[1]
            weights = _get_initializer(graph, weight_name)
            if weights is None:
                raise ValueError(f"Could not find conv weights '{weight_name}'")

            # weights shape: (out_channels, in_channels, kH, kW)
            out_ch, in_ch, kh, kw = weights.shape

            biases = np.zeros(out_ch, dtype=np.float32)
            if len(node.input) > 2:
                b = _get_initializer(graph, node.input[2])
                if b is not None:
                    biases = b

            strides = _get_attr(node, "strides", [1, 1])
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            pad_h = pads[0] if pads else 0
            pad_w = pads[1] if pads else 0

            if len(current_shape) == 3:
                c_in, h_in, w_in = current_shape
            elif len(current_shape) == 1:
                # Flat input reshaped to image
                c_in = in_ch
                total = current_shape[0]
                side = int(math.sqrt(total // c_in))
                h_in, w_in = side, side
            else:
                raise ValueError(f"Cannot determine spatial dims for Conv from shape {current_shape}")

            activation, skip = _peek_activation(nodes, i, node.output[0])
            i += skip

            info = ConvLayerInfo(
                c_in=c_in, h_in=h_in, w_in=w_in,
                num_filters=out_ch, kernel_h=kh, kernel_w=kw,
                stride_h=strides[0], stride_w=strides[1],
                pad_h=pad_h, pad_w=pad_w,
                activation=activation, weights=weights, biases=biases,
            )
            layers.append(info)
            current_shape = (out_ch, info.h_out, info.w_out)

        elif node.op_type in ("MaxPool", "AveragePool"):
            kernel_shape = _get_attr(node, "kernel_shape", [2, 2])
            strides = _get_attr(node, "strides", kernel_shape)

            if len(current_shape) == 3:
                c_in, h_in, w_in = current_shape
            else:
                raise ValueError(f"Pooling requires 3D input, got shape {current_shape}")

            pool_type = "maxpool" if node.op_type == "MaxPool" else "avgpool"
            info = PoolLayerInfo(
                layer_type=pool_type,
                c_in=c_in, h_in=h_in, w_in=w_in,
                kernel_h=kernel_shape[0], kernel_w=kernel_shape[1],
                stride_h=strides[0], stride_w=strides[1],
            )
            layers.append(info)
            current_shape = (c_in, info.h_out, info.w_out)

        elif node.op_type == "GlobalAveragePool":
            if len(current_shape) == 3:
                c_in, h_in, w_in = current_shape
            else:
                raise ValueError(f"GlobalAveragePool requires 3D input")

            info = PoolLayerInfo(
                layer_type="avgpool",
                c_in=c_in, h_in=h_in, w_in=w_in,
                kernel_h=h_in, kernel_w=w_in,
                stride_h=h_in, stride_w=w_in,
            )
            layers.append(info)
            current_shape = (c_in, 1, 1)

        elif node.op_type in ("Flatten", "Reshape"):
            total = 1
            for d in current_shape:
                total *= d
            layers.append(FlattenLayerInfo(num_inputs=total))
            current_shape = (total,)

        elif node.op_type in ("RNN", "LSTM"):
            # ONNX RNN format: inputs are X, W, R, B (optional)
            w_ih = _get_initializer(graph, node.input[1])  # W
            w_hh = _get_initializer(graph, node.input[2])  # R
            biases_rnn = None
            if len(node.input) > 3:
                biases_rnn = _get_initializer(graph, node.input[3])

            hidden_size = _get_attr(node, "hidden_size", w_hh.shape[-1] if w_hh is not None else 0)

            if len(current_shape) == 1:
                input_size = current_shape[0]
            else:
                input_size = current_shape[-1]

            cell_type = "lstm" if node.op_type == "LSTM" else "rnn"
            info = RNNLayerInfo(
                layer_type=cell_type,
                input_size=input_size,
                hidden_size=hidden_size,
                seq_len=1,
                weights_ih=w_ih if w_ih is not None else np.array([]),
                weights_hh=w_hh if w_hh is not None else np.array([]),
                biases=biases_rnn if biases_rnn is not None else np.array([]),
            )
            layers.append(info)
            current_shape = (hidden_size,)

        elif node.op_type in SUPPORTED_ACTIVATIONS:
            pass  # Already consumed by _peek_activation
        elif node.op_type == "Add":
            pass  # Already consumed by MatMul pattern
        elif node.op_type in ("Shape", "Gather", "Unsqueeze", "Concat",
                              "Constant", "Squeeze", "Transpose",
                              "BatchNormalization", "Dropout"):
            pass  # Skip utility ops

        i += 1

    if not layers:
        raise ValueError("No layers found in ONNX model")

    return ParsedNetwork(
        name=network_name, num_inputs=num_inputs,
        input_shape=input_shape, layers=layers,
    )


def float_to_fixed_hex(value: float, int_bits: int, frac_bits: int) -> str:
    """Convert a float to fixed-point hex string."""
    total_bits = int_bits + frac_bits
    scale = 2**frac_bits
    fixed_val = int(round(value * scale))

    # Two's complement for negative values
    if fixed_val < 0:
        fixed_val = (1 << total_bits) + fixed_val

    # Clamp
    max_val = (1 << total_bits) - 1
    fixed_val = max(0, min(fixed_val, max_val))

    hex_digits = (total_bits + 3) // 4
    return f"{fixed_val:0{hex_digits}X}"


def export_weights_hex(
    network: ParsedNetwork,
    output_dir: str | Path,
    int_bits: int,
    frac_bits: int,
) -> dict[str, Path]:
    """
    Export network weights to hex files for BRAM initialization.

    Each layer gets its own file. Weights are stored in the order expected
    by the calculation_unit: for each neuron, all input weights followed by
    the bias, then next neuron.

    Returns dict mapping layer name to output file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    for layer_idx, layer in enumerate(network.layers):
        filename = f"layer_{layer_idx}_weights.hex"
        filepath = output_dir / filename

        lines = []
        # For each output neuron
        for j in range(layer.num_outputs):
            # Weights for this neuron (all inputs)
            for k in range(layer.num_inputs):
                val = float(layer.weights[k, j])
                lines.append(float_to_fixed_hex(val, int_bits, frac_bits))
            # Bias for this neuron
            val = float(layer.biases[j])
            lines.append(float_to_fixed_hex(val, int_bits, frac_bits))

        filepath.write_text("\n".join(lines) + "\n")
        files[f"layer_{layer_idx}"] = filepath

    return files


def export_weights_flat_hex(
    network: ParsedNetwork,
    output_path: str | Path,
    int_bits: int,
    frac_bits: int,
    input_base_addr: int = 0,
    weights_base_addr: int = 256,
) -> Path:
    """
    Export all weights to a single flat hex file matching the BRAM memory map.

    Format matches what the xor_tb testbench loads: sequential weight values
    starting at weights_base_addr.

    Weight order: layer 0 neuron 0 weights + bias, neuron 1 weights + bias, ...
                  layer 1 neuron 0 weights + bias, ...
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for layer in network.layers:
        for j in range(layer.num_outputs):
            for k in range(layer.num_inputs):
                val = float(layer.weights[k, j])
                lines.append(float_to_fixed_hex(val, int_bits, frac_bits))
            val = float(layer.biases[j])
            lines.append(float_to_fixed_hex(val, int_bits, frac_bits))

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def network_to_yaml_layers(network: ParsedNetwork) -> list[dict]:
    """Convert ParsedNetwork layers to YAML-compatible layer dicts."""
    yaml_layers = []
    for layer in network.layers:
        if isinstance(layer, DenseLayerInfo):
            yaml_layers.append({
                "type": "dense",
                "neurons": layer.num_outputs,
                "activation": layer.activation if layer.activation != "none" else "sigmoid",
            })
        elif isinstance(layer, ConvLayerInfo):
            yaml_layers.append({
                "type": "conv2d",
                "filters": layer.num_filters,
                "kernel_size": [layer.kernel_h, layer.kernel_w],
                "stride": [layer.stride_h, layer.stride_w],
                "padding": [layer.pad_h, layer.pad_w],
                "activation": layer.activation,
            })
        elif isinstance(layer, PoolLayerInfo):
            yaml_layers.append({
                "type": layer.layer_type,
                "kernel_size": [layer.kernel_h, layer.kernel_w],
                "stride": [layer.stride_h, layer.stride_w],
            })
        elif isinstance(layer, FlattenLayerInfo):
            yaml_layers.append({"type": "flatten"})
        elif isinstance(layer, RNNLayerInfo):
            yaml_layers.append({
                "type": layer.layer_type,
                "hidden_size": layer.hidden_size,
                "seq_len": layer.seq_len,
            })
    return yaml_layers


def print_network_summary(network: ParsedNetwork) -> None:
    """Print a human-readable summary of the parsed network."""
    print(f"  Network: {network.name}")
    print(f"  Input shape: {network.input_shape}")
    print(f"  Total inputs: {network.num_inputs}")
    print(f"  Layers: {len(network.layers)}")

    total_params = 0
    for i, layer in enumerate(network.layers):
        if isinstance(layer, DenseLayerInfo):
            params = layer.num_inputs * layer.num_outputs + layer.num_outputs
            print(f"    [{i}] Dense: {layer.num_inputs} -> {layer.num_outputs} "
                  f"({layer.activation}, {params} params)")
        elif isinstance(layer, ConvLayerInfo):
            params = (layer.kernel_h * layer.kernel_w * layer.c_in + 1) * layer.num_filters
            print(f"    [{i}] Conv2D: {layer.c_in}x{layer.h_in}x{layer.w_in} -> "
                  f"{layer.num_filters}x{layer.h_out}x{layer.w_out} "
                  f"(k={layer.kernel_h}x{layer.kernel_w}, {layer.activation}, {params} params)")
        elif isinstance(layer, PoolLayerInfo):
            params = 0
            print(f"    [{i}] {layer.layer_type.title()}: "
                  f"{layer.c_in}x{layer.h_in}x{layer.w_in} -> "
                  f"{layer.c_in}x{layer.h_out}x{layer.w_out} "
                  f"(k={layer.kernel_h}x{layer.kernel_w})")
        elif isinstance(layer, FlattenLayerInfo):
            params = 0
            print(f"    [{i}] Flatten: {layer.num_inputs} -> {layer.num_outputs}")
        elif isinstance(layer, RNNLayerInfo):
            if layer.layer_type == "lstm":
                params = 4 * (layer.input_size + layer.hidden_size + 1) * layer.hidden_size
            else:
                params = (layer.input_size + layer.hidden_size + 1) * layer.hidden_size
            print(f"    [{i}] {layer.layer_type.upper()}: in={layer.input_size} "
                  f"hidden={layer.hidden_size} ({params} params)")
        total_params += params
    print(f"  Total parameters: {total_params}")

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
SUPPORTED_ACTIVATIONS = {"Sigmoid", "Relu"}
SUPPORTED_DENSE_OPS = {"Gemm", "MatMul"}
SUPPORTED_OPS = SUPPORTED_ACTIVATIONS | SUPPORTED_DENSE_OPS | {"Add", "Reshape", "Flatten"}


@dataclass
class DenseLayerInfo:
    """Parsed info for a single dense layer."""

    num_inputs: int
    num_outputs: int
    activation: str  # "sigmoid", "relu", or "none"
    weights: np.ndarray  # shape (num_inputs, num_outputs)
    biases: np.ndarray  # shape (num_outputs,)


@dataclass
class ParsedNetwork:
    """Complete parsed network ready for VHDL generation."""

    name: str
    num_inputs: int
    layers: list[DenseLayerInfo] = field(default_factory=list)

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
    return default


def _check_unsupported_ops(graph) -> list[str]:
    """Return list of unsupported op types found in the graph."""
    unsupported = []
    for node in graph.node:
        if node.op_type not in SUPPORTED_OPS:
            unsupported.append(node.op_type)
    return unsupported


def parse_onnx(model_path: str | Path) -> ParsedNetwork:
    """
    Parse an ONNX model file into a ParsedNetwork.

    Supports sequential dense networks with the patterns:
      - Gemm -> Activation
      - MatMul -> Add -> Activation
      - Gemm (no following activation = linear/none)

    Args:
        model_path: Path to .onnx file

    Returns:
        ParsedNetwork with topology and weights

    Raises:
        ImportError: If onnx package not installed
        ValueError: If model contains unsupported ops or topology
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

    # Determine input size from graph inputs (skip initializer names)
    init_names = {init.name for init in graph.initializer}
    real_inputs = [inp for inp in graph.input if inp.name not in init_names]
    if len(real_inputs) != 1:
        raise ValueError(
            f"Expected exactly 1 network input, got {len(real_inputs)}: "
            f"{[i.name for i in real_inputs]}"
        )

    input_shape = real_inputs[0].type.tensor_type.shape.dim
    num_inputs = input_shape[-1].dim_value
    if num_inputs == 0:
        raise ValueError("Input has dynamic size — must be fixed for hardware generation")

    network_name = graph.name or model_path.stem
    layers: list[DenseLayerInfo] = []
    nodes = list(graph.node)

    # Map output_name -> node for lookahead
    output_to_node = {}
    for node in nodes:
        for out in node.output:
            output_to_node[out] = node

    i = 0
    while i < len(nodes):
        node = nodes[i]

        if node.op_type == "Gemm":
            # Gemm: Y = alpha * A * B + beta * C
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None

            weights = _get_initializer(graph, weight_name)
            biases = _get_initializer(graph, bias_name) if bias_name else None

            if weights is None:
                raise ValueError(f"Could not find weights '{weight_name}' in initializers")

            transB = _get_attr(node, "transB", 0)
            if transB:
                weights = weights.T  # Transpose to (in_features, out_features)

            transA = _get_attr(node, "transA", 0)
            if transA:
                raise ValueError("transA=1 on Gemm is not supported")

            num_in = weights.shape[0]
            num_out = weights.shape[1]

            if biases is None:
                biases = np.zeros(num_out, dtype=np.float32)

            # Check if next node is an activation
            activation = "none"
            gemm_output = node.output[0]
            if i + 1 < len(nodes) and nodes[i + 1].op_type in SUPPORTED_ACTIVATIONS:
                next_node = nodes[i + 1]
                if next_node.input[0] == gemm_output:
                    activation = next_node.op_type.lower()
                    i += 1  # Skip the activation node

            layers.append(
                DenseLayerInfo(
                    num_inputs=num_in,
                    num_outputs=num_out,
                    activation=activation,
                    weights=weights,
                    biases=biases,
                )
            )

        elif node.op_type == "MatMul":
            # MatMul + Add pattern (alternative to Gemm)
            weight_name = node.input[1]
            weights = _get_initializer(graph, weight_name)
            if weights is None:
                raise ValueError(f"Could not find weights '{weight_name}' in initializers")

            num_in = weights.shape[0]
            num_out = weights.shape[1]
            matmul_output = node.output[0]

            # Look for Add node
            biases = np.zeros(num_out, dtype=np.float32)
            if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                add_node = nodes[i + 1]
                if add_node.input[0] == matmul_output:
                    bias_name = add_node.input[1]
                    bias_arr = _get_initializer(graph, bias_name)
                    if bias_arr is not None:
                        biases = bias_arr
                    matmul_output = add_node.output[0]
                    i += 1

            # Check for activation
            activation = "none"
            if i + 1 < len(nodes) and nodes[i + 1].op_type in SUPPORTED_ACTIVATIONS:
                next_node = nodes[i + 1]
                if next_node.input[0] == matmul_output:
                    activation = next_node.op_type.lower()
                    i += 1

            layers.append(
                DenseLayerInfo(
                    num_inputs=num_in,
                    num_outputs=num_out,
                    activation=activation,
                    weights=weights,
                    biases=biases,
                )
            )

        elif node.op_type in ("Reshape", "Flatten"):
            # Skip reshape/flatten — they don't map to hardware layers
            pass
        elif node.op_type in SUPPORTED_ACTIVATIONS:
            # Standalone activation not preceded by Gemm/MatMul — skip
            # (it was already consumed by the pattern above, or is standalone)
            pass
        elif node.op_type == "Add":
            # Standalone Add not preceded by MatMul — skip
            pass

        i += 1

    if not layers:
        raise ValueError("No dense layers found in ONNX model")

    return ParsedNetwork(name=network_name, num_inputs=num_inputs, layers=layers)


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
        yaml_layers.append(
            {
                "type": "dense",
                "neurons": layer.num_outputs,
                "activation": layer.activation if layer.activation != "none" else "sigmoid",
            }
        )
    return yaml_layers


def print_network_summary(network: ParsedNetwork) -> None:
    """Print a human-readable summary of the parsed network."""
    print(f"  Network: {network.name}")
    print(f"  Inputs: {network.num_inputs}")
    print(f"  Layers: {len(network.layers)}")
    for i, layer in enumerate(network.layers):
        act = layer.activation
        total_params = layer.num_inputs * layer.num_outputs + layer.num_outputs
        print(
            f"    [{i}] Dense: {layer.num_inputs} -> {layer.num_outputs} "
            f"({act}, {total_params} params)"
        )
    total = sum(
        l.num_inputs * l.num_outputs + l.num_outputs for l in network.layers
    )
    print(f"  Total parameters: {total}")

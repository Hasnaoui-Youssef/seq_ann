"""
YAML Configuration Schema for Neural Network Accelerator

This module loads the JSON schema and provides custom validation functions.
"""

import json
from pathlib import Path


def _load_schema() -> dict:
    """Load the JSON schema from file."""
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, 'r') as f:
        return json.load(f)


CONFIG_SCHEMA = _load_schema()


def is_power_of_two(n: int) -> bool:
    """Check if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def validate_custom_constraints(config: dict) -> list[str]:
    """
    Validate constraints that cannot be expressed in JSON Schema.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check lut_bits > range_bits constraint
    sigmoid = config.get("sigmoid", {})
    lut_bits = sigmoid.get("lut_bits")
    range_bits = sigmoid.get("range_bits")

    if lut_bits is not None and range_bits is not None:
        if lut_bits <= range_bits:
            errors.append(
                f"sigmoid.lut_bits ({lut_bits}) must be greater than "
                f"range_bits ({range_bits}). Increase lut_bits or decrease range_bits."
            )

    # Check total data width is reasonable
    precision = config.get("precision", {})
    int_bits = precision.get("int_bits")
    frac_bits = precision.get("frac_bits")
    total_bits = precision.get("total_bits")

    # Calculate effective values
    if total_bits is not None:
        effective_total = total_bits
    elif int_bits is not None and frac_bits is not None:
        effective_total = int_bits + frac_bits
    else:
        errors.append("precision: must specify either (int_bits + frac_bits) or total_bits")
        effective_total = None

    if effective_total is not None:
        if effective_total > 64:
            errors.append(f"Total data width ({effective_total}) exceeds 64 bits")
        if effective_total not in [8, 16, 32, 64]:
            errors.append(f"Total data width ({effective_total}) should be 8, 16, 32, or 64 for alignment")

    # Check memory addresses do not overlap (basic check)
    memory = config.get("memory", {})
    input_base = memory.get("input_base_addr", 0)
    weights_base = memory.get("weights_base_addr", 256)
    output_base = memory.get("output_base_addr", 61440)

    if input_base >= weights_base:
        errors.append(f"input_base_addr ({input_base}) must be less than weights_base_addr ({weights_base})")
    if weights_base >= output_base:
        errors.append(f"weights_base_addr ({weights_base}) must be less than output_base_addr ({output_base})")

    # Validate network layer sequence
    layers = config.get("network", {}).get("layers", [])
    if layers:
        # Check first layer is not flatten
        if layers[0].get("type") == "flatten":
            errors.append("First layer cannot be 'flatten'")

        # Check flatten is not followed by conv2d/maxpool/avgpool
        for i, layer in enumerate(layers[:-1]):
            if layer.get("type") == "flatten":
                next_type = layers[i + 1].get("type")
                if next_type in ["conv2d", "maxpool", "avgpool"]:
                    errors.append(f"Layer {i + 1}: '{next_type}' cannot follow 'flatten'")

    return errors

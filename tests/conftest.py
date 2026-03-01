"""Shared fixtures and configuration for cocotb tests."""

import os
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# VHDL source files in compilation order (matches make/sources.mk)
VHDL_SOURCES = [
    ROOT / "src" / "packages" / "types.vhd",
    ROOT / "src" / "packages" / "pkg_layer.vhd",
    ROOT / "src" / "packages" / "sigmoid_lut_pkg.vhd",
    ROOT / "src" / "packages" / "layer_interface_pkg.vhd",
    ROOT / "src" / "core" / "accumulator" / "half_adder.vhd",
    ROOT / "src" / "core" / "accumulator" / "signed_mult.vhd",
    ROOT / "src" / "core" / "neuron" / "activation_func.vhd",
    ROOT / "src" / "core" / "accumulator" / "full_adder.vhd",
    ROOT / "src" / "core" / "accumulator" / "n_bit_adder.vhd",
    ROOT / "src" / "core" / "accumulator" / "acc.vhd",
    ROOT / "src" / "core" / "layer" / "weight_bank.vhd",
    ROOT / "src" / "core" / "neuron" / "neuron.vhd",
    ROOT / "src" / "core" / "layer" / "layer.vhd",
    ROOT / "src" / "layers" / "conv2d_layer.vhd",
    ROOT / "src" / "layers" / "maxpool_layer.vhd",
    ROOT / "src" / "layers" / "avgpool_layer.vhd",
    ROOT / "src" / "layers" / "flatten_layer.vhd",
    ROOT / "src" / "layers" / "rnn_cell.vhd",
    ROOT / "src" / "layers" / "lstm_cell.vhd",
    ROOT / "src" / "layers" / "rnn_layer.vhd",
    ROOT / "src" / "memory" / "bram.vhd",
    ROOT / "src" / "units" / "memory_control_unit.vhd",
    ROOT / "src" / "units" / "calculation_unit.vhd",
    ROOT / "src" / "units" / "control_unit.vhd",
    ROOT / "src" / "neural_network.vhd",
]

GHDL_COMPILE_ARGS = ["--std=08", "--ieee=synopsys", "-frelaxed", "--warn-no-vital-generic"]
GHDL_SIM_ARGS = ["--ieee-asserts=disable"]

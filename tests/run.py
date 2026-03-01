"""Pytest runner for cocotb testbenches.

Usage:
    pytest tests/run.py                    # Run all tests
    pytest tests/run.py -k activation      # Run activation tests only
    pytest tests/run.py -k xor             # Run XOR test only
"""

import os
from pathlib import Path

from cocotb_test.simulator import run

# Project root
ROOT = Path(__file__).parent.parent

# VHDL source files in compilation order
VHDL_SOURCES = [str(f) for f in [
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
    ROOT / "tests" / "cocotb_wrappers.vhd",
]]

COMPILE_ARGS = ["--std=08", "--ieee=synopsys", "-frelaxed", "--warn-no-vital-generic"]
SIM_ARGS = ["--ieee-asserts=disable"]

TESTS_DIR = str(ROOT / "tests")
SIM_BUILD = str(ROOT / "sim_build")


def test_activation_func():
    """Test sigmoid activation function."""
    run(
        simulator="ghdl",
        vhdl_sources=VHDL_SOURCES,
        toplevel="activation_func_wrap",
        toplevel_lang="vhdl",
        module="test_activation_func",
        compile_args=COMPILE_ARGS,
        sim_args=SIM_ARGS,
        python_search=[TESTS_DIR],
        sim_build=os.path.join(SIM_BUILD, "activation_func"),
        waves=True,
    )


def test_neuron():
    """Test neuron with forward and backward pass."""
    run(
        simulator="ghdl",
        vhdl_sources=VHDL_SOURCES,
        toplevel="neuron_wrap",
        toplevel_lang="vhdl",
        module="test_neuron",
        compile_args=COMPILE_ARGS,
        sim_args=SIM_ARGS,
        python_search=[TESTS_DIR],
        sim_build=os.path.join(SIM_BUILD, "neuron"),
        waves=True,
    )


def test_layer():
    """Test layer with weight loading and forward pass."""
    run(
        simulator="ghdl",
        vhdl_sources=VHDL_SOURCES,
        toplevel="layer_wrap",
        toplevel_lang="vhdl",
        module="test_layer",
        compile_args=COMPILE_ARGS,
        sim_args=SIM_ARGS,
        python_search=[TESTS_DIR],
        sim_build=os.path.join(SIM_BUILD, "layer"),
        waves=True,
    )


def test_xor():
    """Test XOR inference on the full neural network."""
    run(
        simulator="ghdl",
        vhdl_sources=VHDL_SOURCES,
        toplevel="nn_xor_wrap",
        toplevel_lang="vhdl",
        module="test_xor",
        compile_args=COMPILE_ARGS,
        sim_args=SIM_ARGS,
        python_search=[TESTS_DIR],
        sim_build=os.path.join(SIM_BUILD, "xor"),
        waves=True,
    )

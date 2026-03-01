"""cocotb testbench for the neuron entity (via neuron_wrap)."""

import math
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from sfixed_utils import (
    real_to_sfixed, sfixed_to_real,
    set_sfixed, get_sfixed,
    pack_sfixed_array, unpack_sfixed_array,
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


async def reset(dut, cycles=2):
    dut.rst.value = 1
    dut.fwd_en.value = 0
    dut.bwd_en.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def run_forward(dut, inputs, weights, bias):
    """Run a forward pass and return the output as a float."""
    dut.inputs_flat.value = pack_sfixed_array(inputs)
    dut.weights_flat.value = pack_sfixed_array(weights)
    set_sfixed(dut.bias_slv, bias)

    dut.fwd_en.value = 1
    await RisingEdge(dut.clk)
    # Output registers on the next rising edge after fwd_en
    await RisingEdge(dut.clk)
    dut.fwd_en.value = 0
    await Timer(1, unit="ns")
    return get_sfixed(dut.output_slv)


@cocotb.test()
async def test_neuron_forward_basic(dut):
    """Test neuron forward pass with known inputs/weights."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # NUM_INPUTS = 4 (set in wrapper)
    test_cases = [
        {
            "inputs": [0.5, 0.5, 0.5, 0.5],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "bias": 0.0,
            "desc": "Simple positive values",
        },
        {
            "inputs": [1.0, 1.0, 1.0, 1.0],
            "weights": [0.5, 0.5, 0.5, 0.5],
            "bias": 1.0,
            "desc": "With positive bias",
        },
        {
            "inputs": [-1.0, -1.0, 1.0, 1.0],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "bias": 0.0,
            "desc": "Mixed positive/negative",
        },
        {
            "inputs": [0.0, 0.0, 0.0, 0.0],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "bias": 0.0,
            "desc": "All zero inputs",
        },
    ]

    tolerance = 0.05
    fail_count = 0

    for tc in test_cases:
        output = await run_forward(dut, tc["inputs"], tc["weights"], tc["bias"])
        net_sum = sum(i * w for i, w in zip(tc["inputs"], tc["weights"])) + tc["bias"]
        expected = sigmoid(net_sum)
        error = abs(output - expected)

        if error < tolerance:
            dut._log.info(f"PASS  {tc['desc']:35s}  sum={net_sum:.4f}  sig={expected:.6f}  out={output:.6f}")
        else:
            fail_count += 1
            dut._log.error(f"FAIL  {tc['desc']:35s}  sum={net_sum:.4f}  sig={expected:.6f}  out={output:.6f}  err={error:.6f}")

    assert fail_count == 0, f"{fail_count} forward pass tests failed"


@cocotb.test()
async def test_neuron_backward(dut):
    """Test neuron backward pass produces weight gradients."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    inputs = [0.5, 1.0, -0.5, 0.25]
    weights = [1.0, 0.5, -0.5, 0.75]
    bias = 0.5

    # Run forward first (stores internal state needed for backward)
    await run_forward(dut, inputs, weights, bias)

    # Run backward pass with an error signal
    set_sfixed(dut.error_slv, 0.1)
    dut.bwd_en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.bwd_en.value = 0
    await Timer(1, unit="ns")

    # Check that gradients are non-zero
    grad_weights = unpack_sfixed_array(dut.grad_weights_flat, 4)
    for i, gw in enumerate(grad_weights):
        dut._log.info(f"  grad_weights[{i}] = {gw:.6f}")

    grad_bias = get_sfixed(dut.grad_bias_slv)
    dut._log.info(f"  grad_bias = {grad_bias:.6f}")
    assert grad_bias != 0.0, "Bias gradient should be non-zero for non-zero error"

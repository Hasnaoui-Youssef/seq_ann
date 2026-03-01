"""cocotb testbench for the layer entity (via layer_wrap)."""

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
    dut.fwd_en.value = 1
    dut.bwd_en.value = 0
    dut.weight_load_en.value = 0
    dut.weight_save_en.value = 0
    dut.weight_update_en.value = 0
    dut.fwd_ctrl_in_valid.value = 0
    dut.fwd_ctrl_in_last.value = 0
    dut.bwd_ctrl_in_valid.value = 0
    dut.bwd_ctrl_in_last.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def load_weights(dut, weights: list[float]):
    """Load weights sequentially into the weight bank."""
    for w in weights:
        set_sfixed(dut.weight_load_data, w)
        dut.weight_load_en.value = 1
        await RisingEdge(dut.clk)
    dut.weight_load_en.value = 0
    await RisingEdge(dut.clk)


async def run_forward(dut, inputs: list[float], num_outputs: int) -> list[float]:
    """Trigger forward pass and return outputs."""
    dut.fwd_data_in_flat.value = pack_sfixed_array(inputs)
    dut.fwd_ctrl_in_valid.value = 1
    dut.fwd_ctrl_in_last.value = 1
    await RisingEdge(dut.clk)
    dut.fwd_ctrl_in_valid.value = 0
    dut.fwd_ctrl_in_last.value = 0
    # Wait for pipeline
    for _ in range(3):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    return unpack_sfixed_array(dut.fwd_data_out_flat, num_outputs)


@cocotb.test()
async def test_layer_forward(dut):
    """Test layer forward pass with 4 inputs -> 2 neurons (sigmoid)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    num_inputs = 4
    num_outputs = 2

    # Weight layout: [n0_w0, n0_w1, n0_w2, n0_w3, n0_bias, n1_w0, ..., n1_bias]
    weights = [1.0, 0.5, -0.5, 0.75, 0.5,
               0.5, 1.0, 0.25, -0.5, -0.25]

    await load_weights(dut, weights)

    inputs = [0.5, 1.0, -0.5, 0.25]

    # Calculate expected outputs
    expected = []
    for n in range(num_outputs):
        base = n * (num_inputs + 1)
        net = weights[base + num_inputs]  # bias
        for j in range(num_inputs):
            net += inputs[j] * weights[base + j]
        expected.append(sigmoid(net))

    outputs = await run_forward(dut, inputs, num_outputs)

    tolerance = 0.05
    fail_count = 0
    for i, (out, exp) in enumerate(zip(outputs, expected)):
        error = abs(out - exp)
        if error < tolerance:
            dut._log.info(f"PASS  output[{i}]={out:.6f}  expected={exp:.6f}  err={error:.6f}")
        else:
            fail_count += 1
            dut._log.error(f"FAIL  output[{i}]={out:.6f}  expected={exp:.6f}  err={error:.6f}")

    assert fail_count == 0, f"{fail_count} outputs failed tolerance check"


@cocotb.test()
async def test_layer_weight_loading(dut):
    """Test that weights can be loaded and produce different outputs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    num_inputs = 4
    num_outputs = 2

    inputs = [1.0, 1.0, 1.0, 1.0]

    # Load all-ones weights
    weights_1 = [1.0] * (num_inputs + 1) * num_outputs
    await load_weights(dut, weights_1)
    out_1 = await run_forward(dut, inputs, num_outputs)

    # Reload with different weights
    await reset(dut)
    weights_2 = [0.1] * (num_inputs + 1) * num_outputs
    await load_weights(dut, weights_2)
    out_2 = await run_forward(dut, inputs, num_outputs)

    # Outputs should differ
    assert abs(out_1[0] - out_2[0]) > 0.01, \
        f"Different weights should produce different outputs: {out_1[0]} vs {out_2[0]}"
    dut._log.info(f"Weight reload verified: {out_1[0]:.4f} vs {out_2[0]:.4f}")

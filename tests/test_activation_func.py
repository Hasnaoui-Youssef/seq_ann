"""cocotb testbench for the activation_func entity (sigmoid architecture)."""

import math
import cocotb
from cocotb.triggers import Timer

from sfixed_utils import (
    DATA_WIDTH, FRAC_BITS, INT_BITS,
    real_to_sfixed, sfixed_to_real,
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@cocotb.test()
async def test_sigmoid_basic(dut):
    """Test sigmoid activation across a range of input values."""
    test_inputs = [
        -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 7.99,
    ]

    pass_count = 0
    fail_count = 0
    tolerance = 0.02

    dut._log.info(f"Testing sigmoid with Q{INT_BITS}.{FRAC_BITS} fixed-point ({DATA_WIDTH}-bit)")

    for x in test_inputs:
        expected = sigmoid(x)
        dut.input_i.value = real_to_sfixed(x)
        await Timer(10, unit="ns")

        output_val = sfixed_to_real(dut.output_o.value.to_unsigned())
        error = abs(output_val - expected)

        if error < tolerance:
            pass_count += 1
            dut._log.info(f"PASS  input={x:8.4f}  output={output_val:.6f}  expected={expected:.6f}  err={error:.6f}")
        else:
            fail_count += 1
            dut._log.error(f"FAIL  input={x:8.4f}  output={output_val:.6f}  expected={expected:.6f}  err={error:.6f}")

    dut._log.info(f"Results: {pass_count}/{pass_count + fail_count} passed")
    assert fail_count == 0, f"{fail_count} tests failed"


@cocotb.test()
async def test_sigmoid_saturation(dut):
    """Test that large positive/negative inputs saturate to ~1.0 / ~0.0."""
    # Large positive → ~1.0
    dut.input_i.value = real_to_sfixed(7.0)
    await Timer(10, unit="ns")
    out = sfixed_to_real(dut.output_o.value.to_unsigned())
    assert out > 0.99, f"sigmoid(7.0) should be ~1.0, got {out}"

    # Large negative → ~0.0
    dut.input_i.value = real_to_sfixed(-7.0)
    await Timer(10, unit="ns")
    out = sfixed_to_real(dut.output_o.value.to_unsigned())
    assert out < 0.01, f"sigmoid(-7.0) should be ~0.0, got {out}"


@cocotb.test()
async def test_sigmoid_midpoint(dut):
    """Test that sigmoid(0) ≈ 0.5."""
    dut.input_i.value = real_to_sfixed(0.0)
    await Timer(10, unit="ns")
    out = sfixed_to_real(dut.output_o.value.to_unsigned())
    assert abs(out - 0.5) < 0.01, f"sigmoid(0) should be ~0.5, got {out}"


@cocotb.test()
async def test_sigmoid_symmetry(dut):
    """Test sigmoid(x) + sigmoid(-x) ≈ 1.0 (symmetry property)."""
    test_vals = [0.5, 1.0, 2.0, 3.0]

    for x in test_vals:
        dut.input_i.value = real_to_sfixed(x)
        await Timer(10, unit="ns")
        pos_out = sfixed_to_real(dut.output_o.value.to_unsigned())

        dut.input_i.value = real_to_sfixed(-x)
        await Timer(10, unit="ns")
        neg_out = sfixed_to_real(dut.output_o.value.to_unsigned())

        total = pos_out + neg_out
        assert abs(total - 1.0) < 0.02, f"sigmoid({x}) + sigmoid({-x}) = {total}, expected ~1.0"

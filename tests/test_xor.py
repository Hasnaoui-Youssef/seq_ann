"""cocotb testbench for the neural_network entity — XOR inference test."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from sfixed_utils import (
    real_to_sfixed, sfixed_to_real,
    get_sfixed,
)


async def reset(dut, cycles=2):
    dut.rst.value = 1
    dut.load_weights.value = 0
    dut.start.value = 0
    dut.train_mode.value = 0
    dut.host_write_en.value = 0
    dut.host_addr.value = 0
    dut.host_data.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)


async def write_mem(dut, addr: int, val: float):
    dut.host_addr.value = addr
    dut.host_data.value = real_to_sfixed(val)
    dut.host_write_en.value = 1
    await RisingEdge(dut.clk)
    dut.host_write_en.value = 0
    await RisingEdge(dut.clk)


async def wait_for_signal(dut, signal, value=1, timeout_cycles=200):
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(signal.value) == value:
            return True
    raise TimeoutError(f"Signal did not reach {value} within {timeout_cycles} cycles")


async def run_inference(dut, in1: float, in2: float) -> float:
    """Load inputs, run inference, return output."""
    # Wait for done to deassert if still high
    while int(dut.done.value) == 1:
        await RisingEdge(dut.clk)

    # Load inputs
    await write_mem(dut, 0, in1)
    await write_mem(dut, 1, in2)

    # Wait for ready
    await wait_for_signal(dut, dut.ready)

    # Start inference
    dut.start.value = 1
    dut.train_mode.value = 0
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    await wait_for_signal(dut, dut.done)
    await Timer(1, unit="ns")

    return get_sfixed(dut.output_data_slv)


@cocotb.test()
async def test_xor_inference(dut):
    """Test XOR inference with hand-crafted weights.

    Network: 2 inputs → 3 hidden (sigmoid) → 1 output (sigmoid)
    """
    clock = Clock(dut.clk, 100, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Load weights into memory:
    # Memory layout: inputs at 0-1, target at 2, weights at 3+
    # Layer 0 (3 neurons, 2 inputs + bias each) = 9 weights at addr 3-11
    # N0 (OR-like): w=[10, 10], b=-5
    await write_mem(dut, 3, 10.0)
    await write_mem(dut, 4, 10.0)
    await write_mem(dut, 5, -5.0)
    # N1 (NAND-like): w=[-10, -10], b=15
    await write_mem(dut, 6, -10.0)
    await write_mem(dut, 7, -10.0)
    await write_mem(dut, 8, 15.0)
    # N2 (Unused/Zero): w=[0, 0], b=0
    await write_mem(dut, 9, 0.0)
    await write_mem(dut, 10, 0.0)
    await write_mem(dut, 11, 0.0)

    # Layer 1 (1 neuron, 3 inputs + bias) = 4 weights at addr 12-15
    # N0 (AND-like): w=[10, 10, 0], b=-15
    await write_mem(dut, 12, 10.0)
    await write_mem(dut, 13, 10.0)
    await write_mem(dut, 14, 0.0)
    await write_mem(dut, 15, -15.0)

    dut._log.info("Weights loaded. Triggering weight load...")

    # Trigger weight loading from memory to weight banks
    dut.load_weights.value = 1
    await RisingEdge(dut.clk)
    dut.load_weights.value = 0

    await wait_for_signal(dut, dut.weights_loaded, timeout_cycles=500)
    dut._log.info("Weight banks ready. Running XOR tests...")

    # XOR truth table
    xor_tests = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]

    tolerance = 0.1
    fail_count = 0

    for in1, in2, expected in xor_tests:
        output = await run_inference(dut, in1, in2)
        error = abs(output - expected)

        if error < tolerance:
            dut._log.info(f"PASS  ({in1}, {in2}) → {output:.4f}  expected {expected:.1f}")
        else:
            fail_count += 1
            dut._log.error(f"FAIL  ({in1}, {in2}) → {output:.4f}  expected {expected:.1f}  err={error:.4f}")

    assert fail_count == 0, f"{fail_count}/4 XOR tests failed"
    dut._log.info("All XOR tests passed!")

# Hardware-Accelerated Neural Network in VHDL - AI Agent Guidelines

This document outlines the key design principles and guidelines for assisting with the development of a hardware-accelerated artificial neural network implementation in VHDL.

## Fixed-Point Arithmetic

### Core Principles
- **All operations use fixed-point arithmetic** with a consistent fractional bit width across the design
- **Accumulators, multipliers, and any resizing operations** maintain the same number of fractional bits as defined in `types.vhd`
- **The `types.vhd` package** serves as the central configuration point for:
  - Total data width
  - Number of fractional bits
- **Type conversion convention**: When converting `std_logic_vector` to `sfixed`:
  - The lower N bits (where N = number of fractional bits) are interpreted as the fractional part
  - No additional scaling or shifting is applied during conversion
- **Entity usage** We are aiming to create a modular project, therefore:
  - If we are going to use a logic block, we first check if a component doing that already exists.
  - We try and make a separation of concerns, for example, a neuron will always contain an accumulator and an activation function.
  - If things are broken within a component, we fix that, or we create a new architecture to accommodate for our different use case, but architecture creation should be a last resort and you are to prompt me if doing so explaining in as much details as possible what's wrong with the current one and why fixing it alone won't work.

### Implementation Notes
- Maintain fractional bit alignment throughout arithmetic operations
- Plan for potential overflow/underflow in accumulator sizing
- Document any deviation from standard fractional bit width with clear justification

## Testing and Simulation

### Testbench Output Management
- **All simulation reports must be saved to log files**
- **Create a dedicated `log/` folder** for storing all test outputs
- Capture and redirect VHDL `report` statements to organized log files
- Use clear, descriptive naming conventions for log files (e.g., `layer1_test_YYYYMMDD.log`)

### Simulation File Format
- **Always use `.ghw` format** for simulation waveform files
- **Never use `.vcd` format**
- This ensures compatibility with our toolchain and provides better debugging capabilities

## Hardware Design Philosophy

### Thinking Close to the Hardware
- **This is a hardware system, not software** - avoid procedural thinking
- Consider:
  - What signals need to be registered vs. combinational?
  - What requires clock edges vs. continuous assignment?
  - Resource utilization (LUTs, registers, DSP blocks)
  - Timing constraints and critical paths
  - Pipeline stages and latency requirements

### Clocking Strategy

#### Clocked Elements (Registered):
- **Neuron internal state** - each neuron contains a register to hold its computed value
- **Layer inputs/outputs** - all data entering and exiting layers is registered
- **Neural network top-level I/O** - inputs and outputs are clocked at the network boundary

#### Combinational Paths (Unclocked):
- **Neuron-to-layer communication** - neurons pass their registered output values to layers **in the same cycle**
- This means:
  - Neurons compute their activation on one clock edge
  - Store the result in an internal register
  - Output this registered value combinationally (without additional clocking) to the layer
  - The layer captures this value on the next clock edge

#### Timing Diagram Example:
```
Clock:     __|‾‾|__|‾‾|__|‾‾|__

Neuron:    [Compute] → [Hold Value]
                          ↓ (combinational)
Layer:              [Capture Input] → [Process]
```

### Pipeline Considerations
- Each neuron acts as a register stage
- Layer boundaries introduce clock cycles
- Plan data flow to minimize unnecessary register stages while maintaining timing

## Code Generation Scripts (Python)

When generating VHDL components via Python scripts:
- **Lookup tables** (sigmoid, tanh) should be generated with proper fixed-point scaling
- **Testbenches** should automatically include log file generation
- **Architecture generation** should respect the clocking guidelines outlined above
- Ensure generated code includes appropriate comments documenting:
  - Fixed-point representation
  - Clock domain information
  - Any assumptions made during generation

## General Best Practices

- Always specify bit widths explicitly
- Use descriptive signal names that indicate their clock domain
- Comment complex fixed-point conversions
- Document the latency of each pipeline stage
- Verify timing closure for critical paths
- Consider synthesis constraints early in design

---

**Remember**: Hardware accelerators gain their speed advantage through parallelism and pipelining, not sequential execution. Design with spatial logic in mind, not temporal control flow.

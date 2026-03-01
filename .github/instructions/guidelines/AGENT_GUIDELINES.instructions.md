# Project Guidelines for AI/Agent Contributors

## Project Overview

This is a hardware neural network accelerator project targeting FPGA/ASIC implementation.
The goal is to create a flexible, composable architecture for image classification and
general neural network inference/training.

---

## Design Rules

### 1. Configuration via Python

All major configuration parameters MUST be set via Python scripts that generate VHDL packages.

**Configurable parameters include:**
- Fixed-point precision (INT_BITS, FRAC_BITS)
- Network topology (layer sizes, types)
- Parallelism levels (e.g., kernel positions in Conv2D)
- Batch size
- Memory map addresses
- LUT sizes (e.g., sigmoid)

**Rationale:** Hardware is generated at compile time, not configured at runtime. This keeps
the hardware simple and allows optimization for specific configurations.

### 2. Fixed-Point Arithmetic

- All computation uses fixed-point arithmetic (sfixed from IEEE fixed_pkg)
- Precision is configured via Python scripts
- The fractional width MUST remain constant throughout the network (no implicit precision changes)
- When multiplying, resize back to original precision immediately
- Use `resize()` with appropriate `round_style` and `overflow_style`

### 3. Separation of Concerns

| Unit | Responsibility | Does NOT do |
|------|---------------|-------------|
| Memory Control Unit | Data movement (load/store) | Computation, weight updates |
| Weight Bank | Weight storage, gradient accumulation, weight updates | Memory access |
| Compute Units (MAC, neurons) | Pure computation | Memory access, control |
| Control Unit | Orchestration, state machine | Computation, data storage |
| Calculation Unit | Forward/backward pass coordination | Memory fetching (future) |

### 4. Weight Management

- Weights are stored in **registers** inside weight banks (not fetched from memory during compute)
- Weight loading from memory is a **separate operation** from inference
- Weight updates happen **in-place** in registers during training
- Memory writes for weights only on **explicit save/checkpoint command**
- Weight sharing (for CNNs) is handled by the weight bank broadcasting to compute units

### 5. Layer Interface

All layer types MUST implement a common interface for composability:
- Forward data path (data_in, valid_in, data_out, valid_out)
- Backward gradient path (grad_in, valid_in, grad_out, valid_out)
- Weight management (load_en, load_data, save_en, save_data)
- Configuration port

### 6. Future SoC Integration

This accelerator is designed to be part of a larger SoC with:
- Network-on-Chip (NoC) for communication
- Multiple CPUs
- Shared memory
- Other accelerators

**Implications:**
- Memory interface should be abstractable
- Control interface should support external commands
- Design for eventual AXI or custom NoC protocol

---

## Coding Standards

### VHDL Style

1. Use VHDL-2008 features (IEEE fixed_pkg, unconstrained arrays)
2. Entity names: `snake_case`
3. Signal names: `snake_case`
4. Constants: `UPPER_SNAKE_CASE`
5. Generics: `UPPER_SNAKE_CASE`
6. Architecture names: typically `rtl` or descriptive like `behavioral`

### File Organization

```
src/
├── packages/          # Type definitions, constants
├── core/              # Basic building blocks (MAC, activations)
├── layers/            # Layer implementations
├── control/           # Control and orchestration
├── memory/            # Memory interfaces
└── top/               # Top-level entities

testbench/             # All testbenches
scripts/               # Python generation scripts
```

### Testing

1. Every entity should have a testbench
2. Use the Makefile: `make TESTBENCH=<name> test`
3. Run all tests before committing: `make test-all`
4. Testbenches should use `std.env.stop` to end simulation

---

## Build System

### Current Tools

- **GHDL**: VHDL simulator (VHDL-2008)
- **GTKWave**: Waveform viewer
- **Python 3**: Configuration scripts, testbench generation
- **Make**: Build orchestration

### Key Commands

```bash
# Activate virtual environment (required for Python scripts)
source .venv/bin/activate

# Run specific testbench
make TESTBENCH=<name> test

# Run all tests
make test-all

# Generate sigmoid LUT and testbenches
make generate

# View waveforms
make TESTBENCH=<name> view
```

### Python Scripts

Located in `scripts/`:
- `generate_sigmoid.py`: Generates sigmoid LUT package
- `generate_testbenches.py`: Generates testbenches from templates
- `config.py`: Central configuration (precision, ranges, etc.)

**Important:** Python scripts enforce constraints (e.g., power-of-2 for LUT sizes).
These constraints exist for hardware efficiency reasons.

---

## Architecture Notes

### Current State

- Basic dense layer network working (XOR demo)
- Sigmoid activation via LUT
- Sequential weight/input loading from memory
- Single-sample inference

### Immediate Goals (Phase 1)

1. Decouple weight loading from inference
2. Implement weight bank entity
3. Weights loaded once, multiple inferences without reload

### Future Goals

- Convolutional layers with weight sharing
- Pooling layers
- Full CNN for image classification
- Training with backpropagation
- SoC integration with NoC

---

## Common Pitfalls

1. **Signal timing**: Remember that signal assignments in a process take effect on the NEXT clock edge. Use variables for immediate values within a process.

2. **Reset handling**: Ensure all state machines and registers have proper reset behavior, and that reset propagation is taken into consideration.

3. **Metavalue warnings**: Initialize all signals. Use `(others => '0')` for vectors.

4. **Fixed-point overflow**: Always check overflow behavior when resizing. Use `fixed_saturate` if saturation is needed.

5. **Memory timing**: BRAM reads have 1-cycle latency. Account for this in state machines.

6. **Testbench synchronization**: Use proper handshaking (wait for signals to stabilize, check both edges of done signals).

---

## Contact / Resources

- Architecture plan: `.github/instructions/architecture/overall_architecture.instructions.md`
- This file: `.github/instructions/guidelines/AGENT_GUIDELINES.md`
- Test with: `make test-all`

# Neural Network Accelerator Architecture Plan

## Project Goal

Create a flexible hardware neural network accelerator capable of:
- Image classification (primary target)
- Composable layer architectures (Conv, Pool, Dense, Flatten, etc.)
- Training with backpropagation
- Efficient weight management with sharing support (for CNNs)

---

## Current State

### What We Have
- Basic neuron with MAC + sigmoid activation
- Layer abstraction (array of neurons)
- Calculation unit (orchestrates forward pass)
- Memory control unit (BRAM interface)
- Control unit (state machine for inference/training)
- Working XOR test demonstrating end-to-end inference

### Current Issues
1. **No layer type abstraction** - only dense layers
2. **Memory fetching in calculation unit** - wrong responsibility
3. **Neurons tightly coupled to memory** - limits flexibility
4. **No support for weight sharing** - blocks CNN implementation

---

## Target Architecture

### High-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         TOP LEVEL                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │   Control   │────▶│   Memory    │────▶│   Layer Array       │ │
│  │    Unit     │     │   Control   │     │  ┌───┐ ┌───┐ ┌───┐  │ │
│  │             │◀────│    Unit     │◀────│  │L0 │─│L1 │─│LN │  │ │
│  └─────────────┘     └─────────────┘     │  └───┘ └───┘ └───┘  │ │
│        │                   │             └─────────────────────┘ │
│        ▼                   ▼                                     │
│  ┌─────────────────────────────────────┐                        │
│  │         Host Interface              │                        │
│  │   (Weight load, Input load, Config) │                        │
│  └─────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

### Layer Architecture (Unified Interface)

Every layer type implements a common interface:

```vhdl
entity layer_interface is
    port (
        clk, rst : in std_logic;

        -- Forward path
        fwd_data_in    : in  data_bus_t;
        fwd_valid_in   : in  std_logic;
        fwd_data_out   : out data_bus_t;
        fwd_valid_out  : out std_logic;

        -- Backward path (training)
        bwd_grad_in    : in  data_bus_t;
        bwd_valid_in   : in  std_logic;
        bwd_grad_out   : out data_bus_t;
        bwd_valid_out  : out std_logic;

        -- Weight management
        weight_load_en   : in  std_logic;
        weight_load_data : in  std_logic_vector;
        weight_load_done : out std_logic;

        weight_save_en   : in  std_logic;
        weight_save_data : out std_logic_vector;
        weight_save_done : out std_logic;

        -- Configuration (set once)
        config_data : in  layer_config_t;
        config_valid: in  std_logic
    );
end entity;
```

### Layer Types to Implement

| Layer Type | Weights | Weight Sharing | Backprop Complexity |
|------------|---------|----------------|---------------------|
| Dense      | Yes     | No             | Medium              |
| Conv2D     | Yes     | Yes (kernel)   | High                |
| MaxPool    | No      | N/A            | Low (index routing) |
| AvgPool    | No      | N/A            | Low                 |
| Flatten    | No      | N/A            | None (reshape)      |
| BatchNorm  | Yes     | No             | Medium              |
| Dropout    | No      | N/A            | None (mask)         |
| ReLU/Sigmoid| No     | N/A            | Low                 |

---

## Weight Bank Design

### Concept

Each layer with weights has a **Weight Bank** that:
- Stores weights in registers (fast access during compute)
- Handles weight loading from memory (one-time or on command)
- Supports weight sharing (same weights → multiple compute units)
- Accumulates gradients during backprop
- Applies weight updates in-place

### Weight Bank Interface

```vhdl
entity weight_bank is
    generic (
        NUM_WEIGHTS   : integer;
        WEIGHT_WIDTH  : integer;
        SHARING_MODE  : sharing_mode_t  -- NONE, SPATIAL (CNN), TEMPORAL (RNN)
    );
    port (
        clk, rst : in std_logic;

        -- Load weights from memory
        load_en   : in  std_logic;
        load_data : in  std_logic_vector(WEIGHT_WIDTH-1 downto 0);
        load_idx  : in  integer range 0 to NUM_WEIGHTS-1;
        load_done : out std_logic;

        -- Read weights for forward pass (parallel access)
        read_en   : in  std_logic;
        read_data : out weight_array_t;  -- All weights available simultaneously

        -- Gradient accumulation (backprop)
        grad_en   : in  std_logic;
        grad_data : in  weight_array_t;
        grad_idx  : in  integer;  -- For shared weights: which instance

        -- Weight update (after accumulation)
        update_en : in  std_logic;
        learn_rate: in  std_logic_vector(WEIGHT_WIDTH-1 downto 0);

        -- Save weights to memory
        save_en   : in  std_logic;
        save_data : out std_logic_vector(WEIGHT_WIDTH-1 downto 0);
        save_idx  : in  integer range 0 to NUM_WEIGHTS-1;
        save_done : out std_logic
    );
end entity;
```

### Weight Sharing for CNNs

```
Conv2D Layer with 3x3 kernel, 8 output channels:

Weight Bank: 3*3*8 = 72 weights (stored once)

        ┌─────────────────────────────────┐
        │         Weight Bank             │
        │  [k0_0..k0_8] [k1_0..k1_8] ...  │
        └───────────────┬─────────────────┘
                        │ broadcast
        ┌───────┬───────┼───────┬───────┐
        ▼       ▼       ▼       ▼       ▼
    ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
    │Pos 0 ││Pos 1 ││Pos 2 ││ ...  ││Pos N │  (spatial positions)
    │MAC   ││MAC   ││MAC   ││      ││MAC   │
    └──────┘└──────┘└──────┘└──────┘└──────┘
        │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼
    grad_0  grad_1  grad_2   ...   grad_N
        │       │       │       │       │
        └───────┴───────┴───────┴───────┘
                        │
                        ▼
              Gradient Accumulator
              (sum all positions)
                        │
                        ▼
                Weight Update
```

---

## Memory Control Unit Redesign

### Responsibilities

1. **Weight Management**
   - Load weights from BRAM to layer weight banks (on init/command)
   - Save weights from weight banks to BRAM (checkpoint/end of training)

2. **Input Management**
   - Load input data from BRAM to first layer
   - For images: handle 2D→1D addressing

3. **Output Management**
   - Store final layer output to BRAM
   - Store intermediate activations (if needed for backprop)

4. **Gradient Management** (training)
   - Route gradients between layers
   - NOT responsible for weight updates (that's in weight bank)

### Memory Map

```
Address Range        | Content
---------------------|---------------------------
0x0000 - 0x00FF     | Input buffer (configurable size)
0x0100 - 0x0FFF     | Layer 0 weights
0x1000 - 0x1FFF     | Layer 1 weights
...                  | ...
0xF000 - 0xF0FF     | Output buffer
0xF100 - 0xFFFF     | Scratch/activations (training)
```

---

## Control Unit Redesign

### States

```
IDLE
  │
  ├──[load_weights]──▶ LOAD_WEIGHTS ──▶ IDLE
  │
  ├──[load_input]────▶ LOAD_INPUT ────▶ IDLE
  │
  ├──[inference]─────▶ RUN_FORWARD ───▶ DONE ──▶ IDLE
  │
  └──[train]─────────▶ RUN_FORWARD ───▶ RUN_BACKWARD ───▶ UPDATE_WEIGHTS ───▶ DONE ──▶ IDLE
```

### Commands (from host)

| Command        | Action                                      |
|----------------|---------------------------------------------|
| LOAD_WEIGHTS   | Memory → Weight Banks (all layers)          |
| LOAD_INPUT     | Memory → First layer input registers        |
| INFERENCE      | Run forward pass, output available          |
| TRAIN_STEP     | Forward → Backward → Weight update          |
| SAVE_WEIGHTS   | Weight Banks → Memory                       |
| CHECKPOINT     | Save current weights to memory              |

---

## Implementation Phases

### Phase 1: Weight Bank & Decoupling (Current Focus)

**Goal**: Separate weight loading from inference

**Tasks**:
1. [ ] Create `weight_bank` entity
2. [ ] Modify `layer` to use weight bank instead of direct weight ports
3. [ ] Modify `neuron` to read from weight bank
4. [ ] Update `memory_control_unit` to handle weight loading
5. [ ] Update `control_unit` with LOAD_WEIGHTS state
6. [ ] Update `calculation_unit` to only handle forward pass orchestration
7. [ ] Test: Load weights once, run multiple inferences

**Success Criteria**:
- Weights loaded once after reset
- Multiple inferences without reloading weights
- XOR test still passes

### Phase 2: Unified Layer Interface

**Goal**: Abstract layer interface for composability

**Tasks**:
1. [ ] Define `layer_interface` package with types
2. [ ] Refactor `dense_layer` to implement interface
3. [ ] Create layer wrapper/adapter if needed
4. [ ] Test layer chaining with new interface

### Phase 3: Activation Functions as Layers

**Goal**: Modular activation functions

**Tasks**:
1. [ ] Create `activation_layer` entity (passthrough with function)
2. [ ] Support ReLU, Sigmoid, Tanh, Softmax
3. [ ] Remove activation from neuron (neuron = pure MAC)
4. [ ] Update layer composition

### Phase 4: Convolutional Layer

**Goal**: Implement Conv2D with weight sharing

**Tasks**:
1. [ ] Design Conv2D architecture with shared weight bank
2. [ ] Implement sliding window input handling
3. [ ] Implement parallel MAC array for kernel
4. [ ] Implement gradient accumulation for shared weights
5. [ ] Test on simple image filter

### Phase 5: Pooling Layers

**Goal**: Implement pooling operations

**Tasks**:
1. [ ] MaxPool2D with index tracking (for backprop)
2. [ ] AvgPool2D
3. [ ] Global pooling variants

### Phase 6: Full CNN Pipeline

**Goal**: LeNet-style network for MNIST

**Tasks**:
1. [ ] Conv → Pool → Conv → Pool → Flatten → Dense → Dense
2. [ ] End-to-end forward pass
3. [ ] End-to-end training
4. [ ] MNIST digit classification demo

---

## Design Rules & Decisions

1. **Parallelism level for Conv2D**:
   - Configurable number of kernel positions executed in parallel
   - Configured via Python scripts at generation time
   - Allows trade-off between area and throughput per deployment target

2. **Batch support**:
   - Mini-batch support will be implemented
   - Enables efficient training with gradient averaging
   - Batch size configurable

3. **Fixed-point precision**:
   - Configurable via Python scripts
   - INT_BITS and FRAC_BITS set at generation time

4. **Memory interface**:
   - Simple BRAM for now
   - Redesign for SoC integration with NoC later
   - Will need to interface with shared memory via Network-on-Chip
   - Target: Coordination between CPUs, ANN accelerators, and shared memory

### Design Principles

1. **Python-driven configuration**: All configurable parameters (parallelism, precision, network topology) are set via Python scripts that generate VHDL packages

2. **Hardware generation, not runtime configuration**: Major architectural choices are compile-time, not runtime configurable (keeps hardware simple)

3. **SoC-ready design**: Architecture should anticipate integration into larger system with NoC

4. **Separation of concerns**:
   - Memory unit: data movement only
   - Weight bank: weight storage and updates
   - Compute units: pure computation
   - Control unit: orchestration
---

## File Structure (Proposed)

```text
src/
├── packages/
│   ├── types.vhd              # Basic types
│   ├── layer_pkg.vhd          # Layer interface definitions
│   └── nn_config_pkg.vhd      # Network configuration types
├── core/
│   ├── accumulator/
│   ├── layer/
│   ├── neuron/
│   └── sigmoid
├── layers/
│   ├── dense_layer.vhd
│   ├── conv2d_layer.vhd
│   ├── maxpool_layer.vhd
│   ├── avgpool_layer.vhd
│   └── flatten_layer.vhd
├── control/
│   ├── control_unit.vhd
│   ├── memory_control_unit.vhd
│   └── calculation_unit.vhd
├── memory/
│   └── bram.vhd
└── top/
    └── neural_network.vhd
```

# Changelog

## [2025-12-17] - Metavalue Warning Fixes

### Fixed
- BRAM: Initialize RAM contents to zeros and use registered outputs
- calculation_unit: Initialize `mem_read_addr`, `mem_update_addr`, `mem_update_grad` on reset
- neuron: Initialize `grad_weights_o`, `grad_inputs_o`, and `stored_inputs` on reset
- memory_control_unit: Address signals already had proper initialization

### Notes
- Remaining warnings at `@0ms` are unavoidable (combinational logic before first clock)
- Warnings at `@12550ns` during first inference are benign (sigmoid LUT indexing during signal transition)

---

## [2025-06-08] - Weight Bank Implementation (Phase 1)

### Added
- **Weight Bank Entity** (`src/weight_bank.vhd`)
  - Stores weights in registers for fast parallel access during inference
  - Sequential loading from memory (one weight per clock)
  - Parallel read access to all weights for compute units
  - Gradient update interface for training (w = w - lr * grad)
  - Save interface for checkpointing weights to memory

- **Decoupled weight loading from inference**
  - `load_weights` signal triggers one-time weight loading from memory
  - `weights_loaded` status indicates weight banks are ready
  - Inference only fetches inputs, uses pre-loaded weights
  - Multiple inferences can run without reloading weights

### Changed
- **Layer Entity** (`src/layer.vhd`)
  - Now contains internal weight bank instead of external weight input
  - New ports: `weight_load_en`, `weight_load_data`, `weight_load_done`
  - New ports: `weight_save_en`, `weight_save_data`, `weight_save_done`
  - New ports: `weight_update_en`, `weight_learn_rate`, `weight_update_done`

- **Calculation Unit** (`src/calculation_unit.vhd`)
  - New `load_weights` input to trigger weight loading
  - New `weights_loaded` output status
  - State machine refactored: separate states for weight loading and inference
  - Weight loading iterates through layers, sending weights sequentially

- **Neural Network** (`src/neural_network.vhd`)
  - New `load_weights` input port
  - New `weights_loaded` output port

- **XOR Testbench** (`testbench/xor_tb.vhd`)
  - Updated to use new weight loading flow
  - Loads weights to memory, triggers `load_weights`, waits for `weights_loaded`

- **Layer Testbench Generator** (`scripts/gen_testbenches.py`)
  - Updated to generate testbenches compatible with new weight bank interface

### Performance
- Inference time reduced from ~9000ns to ~2500ns per sample (XOR test)
- Weights loaded once (~8000ns), then reused for all inferences

---

## [2025-06-08] - YAML Configuration System

### Added
- **YAML-based configuration system** for neural network parameters
  - `scripts/config/` package with schema validation
  - `scripts/config/schema.json` - JSON Schema definition
  - `scripts/config/schema.py` - Schema loading and custom validation
  - `scripts/config/loader.py` - YAML parsing and auto-discovery
  - `configs/default.nn_conf.yaml` - Default configuration
  - `configs/xor.nn_conf.yaml` - XOR network configuration

- **Configuration auto-discovery**: Scripts automatically find `*.nn_conf.yaml` files in project root, falling back to `configs/default.nn_conf.yaml`

- **Schema validation** with custom constraints:
  - Power-of-two validation for LUT size and range
  - Precision can be specified as `int_bits` + `frac_bits` OR `total_bits` (split evenly)
  - Layer definitions with type-specific properties

### Changed
- `scripts/gen_sigmoid_pkg.py` - Now uses YAML config instead of CLI args
- `scripts/gen_testbenches.py` - Now uses YAML config instead of CLI args
- `make/scripts.mk` - Added `CONFIG` variable for explicit config selection
- Sigmoid LUT generation uses efficient bit-manipulation indexing:
  - Flip MSB to transform `[-2^n, 2^n)` to `[0, 2^(n+1))`
  - Direct bit extraction for index (no arithmetic)
  - Saturation-based overflow handling with edge values 0/1 in LUT

### Deprecated
- `scripts/config.py` - Old config class, kept for `gen_neural_network.py` compatibility

### Usage
```bash
# Auto-detect config
make generate-pkg
make generate-tb

# Explicit config
make CONFIG=configs/xor.nn_conf.yaml generate-pkg
```

### Configuration Format
```yaml
network:
  name: "my_network"

precision:
  int_bits: 16
  frac_bits: 16

sigmoid:
  lut_bits: 8      # LUT size = 2^8 = 256
  range_bits: 3    # Range = [-2^3, 2^3) = [-8, 8)

layers:
  - name: "hidden"
    type: "dense"
    inputs: 2
    outputs: 4
    activation: "sigmoid"
```

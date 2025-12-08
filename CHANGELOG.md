# Changelog

## [2025-12-08] - YAML Configuration System

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

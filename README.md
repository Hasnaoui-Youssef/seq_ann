# Neural Network Accelerator (seq_acc)

VHDL-based neural network accelerator targeting FPGA/ASIC, with YAML-driven configuration and ONNX model import.

## Features

- **Hardware Components:**
  - Sigmoid activation function with LUT
  - Configurable neurons with weight banks
  - Dense layer with parallel neuron execution
  - Conv2D, MaxPool, AvgPool, Flatten, RNN, LSTM layer stubs
  - Fixed-point arithmetic (configurable Q-format, default Q16.16)

- **Software Tools:**
  - YAML-driven VHDL package generation (`configure.py`)
  - ONNX model import with weight extraction (`onnx_parser.py`)
  - cocotb-based Python test framework

## Quick Start

### Prerequisites

- Python 3.10+
- GHDL (VHDL-2008 simulator)
- GTKWave (optional, waveform viewer)

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure & Test

```bash
# Generate VHDL packages from default config
make configure

# Run all tests (cocotb + GHDL)
make test

# Run a specific test
python -m pytest tests/run.py -k activation -v
```

### ONNX Model Import

```bash
python scripts/configure.py --model path/to/model.onnx
```

## Project Structure

```
seq_acc/
├── src/                    # VHDL source files
│   ├── packages/           # Type definitions, sigmoid LUT
│   ├── core/               # Core building blocks
│   │   ├── accumulator/    # Adders, multipliers
│   │   ├── neuron/         # Neuron, activation functions
│   │   └── layer/          # Layer, weight bank
│   ├── layers/             # Layer implementations (conv2d, rnn, etc.)
│   ├── memory/             # BRAM interface
│   └── units/              # Control, calculation, memory control units
├── tests/                  # cocotb Python testbenches
│   ├── run.py              # pytest runner
│   ├── cocotb_wrappers.vhd # VHDL wrappers for VPI access
│   └── sfixed_utils.py     # Fixed-point conversion helpers
├── scripts/                # Python generation & tooling
│   ├── configure.py        # Unified config entry point
│   ├── config/             # YAML config loading & schema
│   ├── onnx_parser.py      # ONNX model parser
│   └── gen_sigmoid_pkg.py  # Sigmoid LUT generator
├── configs/                # YAML configuration files
├── data/                   # Generated weight/hex files
├── Makefile                # Build automation
└── requirements.txt        # Python dependencies
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `configure` | Generate VHDL packages from YAML config |
| `test` | Configure + run all cocotb tests |
| `cocotb-test` | Run cocotb tests only |
| `clean` | Remove build artifacts |
| `mnist-harness` | Run MNIST test harness |
| `timeseries-harness` | Run timeseries test harness |

## Configuration

Edit `configs/default.nn_conf.yaml` or create a new `.nn_conf.yaml`:

```yaml
precision:
  int_bits: 16
  frac_bits: 16

sigmoid:
  lut_bits: 8       # 256 LUT entries
  range_bits: 3     # [-8, 8)

hardware:
  acc_size: 4
  num_accs: 2
  num_mults: 8

network:
  name: my_network
  layers:
    - type: dense
      neurons: 3
      activation: sigmoid
    - type: dense
      neurons: 1
      activation: sigmoid
```

## Testing

All tests use [cocotb](https://www.cocotb.org/) with GHDL's VPI interface. cocotb lets us write testbenches in Python instead of VHDL, making it easier to compute expected values, parameterize tests, and integrate with pytest.

GHDL's VPI does not support indexing into `sfixed` array ports directly, so VHDL wrapper entities in `tests/cocotb_wrappers.vhd` flatten array ports into wide `std_logic_vector` signals. The Python helpers in `tests/sfixed_utils.py` handle packing/unpacking between Python floats and the flat bit vectors.

### Test suite

| Test | DUT | What it covers |
|------|-----|----------------|
| `test_activation_func` | `activation_func` (sigmoid) | LUT accuracy, saturation, midpoint, symmetry |
| `test_neuron` | `neuron` | Forward pass, backward pass gradients |
| `test_layer` | `layer` | Forward pass with weight bank, weight reload |
| `test_xor` | `neural_network` | End-to-end XOR inference (2→3→1) |

### Running tests

```bash
make test                                    # Configure + run all
make cocotb-test                             # Run without reconfiguring
python -m pytest tests/run.py -k neuron -v   # Run a specific test
python -m pytest tests/run.py -v             # Verbose output
```

## License

Educational project - use freely.

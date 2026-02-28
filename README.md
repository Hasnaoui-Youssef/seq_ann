# Neural Network Accelerator (seq_acc)

VHDL-based neural network inference accelerator with Keras/TensorFlow integration for weight loading and testing.

**NOTE : This README is deprecated and needs updating to the overall structure of the project!**

## Features

- **Hardware Components:**
  - Sigmoid activation function with LUT
  - Configurable neurons with weight storage
  - Layer composition with parallel neuron execution
  - Fixed-point arithmetic (32-bit data, 16-bit fractional)

- **Software Tools:**
  - Automated VHDL code generation
  - Keras/TensorFlow model integration
  - Weight extraction and fixed-point conversion
  - Testbench generation and validation

## Quick Start

### Prerequisites

- Python 3.8+
- GHDL (VHDL simulator)
- GTKWave (waveform viewer)

### Setup

1. **Create Python virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate VHDL code:**
   ```bash
   make generate
   ```

### Usage

**Run individual testbenches:**
```bash
make TESTBENCH=activation_func  # Test sigmoid activation
make TESTBENCH=neuron           # Test single neuron
make TESTBENCH=layer            # Test layer of neurons
```

**Run all tests:**
```bash
make test-all
```

**Generate neural network testbench:**
```bash
# Use virtual environment
.venv/bin/python scripts/gen_neural_network.py --create-model --layers 4,3,2

# With saved Keras model
.venv/bin/python scripts/gen_neural_network.py --model-file my_model.h5
```

## Project Structure

```
seq_acc/
├── src/                    # VHDL source files
│   ├── packages            # Common type definitions
│   ├── core                # Core logic
│   │   ├──accumulator/     # Accumulator implementation
│   │   ├──layer/           # Neural Network layer
│   │   ├──neuron/          # Neuron implementation
│   │   └──sigmoid/         # **DEPRECATED** placement for the sigmoid lut package (see package folder)
│   ├── memory              # Memory interface
│   ├── units               # Processing units
│   └── top                 # Top level design
├── testbench/              # VHDL testbenches
├── scripts/                # Python generation scripts
├── .venv/                  # Python virtual environment
├── Makefile                # Build automation
└── requirements.txt        # Python dependencies
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `generate` | Generate all VHDL code (LUT + testbenches) |
| `generate-pkg` | Generate sigmoid LUT package only |
| `generate-tb` | Generate testbenches only |
| `generate-nn-tb` | Generate neural network testbench from Keras model |
| `make` | Compile VHDL (requires `TESTBENCH=name`) |
| `run` | Run simulation |
| `view` | Open GTKWave viewer |
| `test-all` | Run all testbenches sequentially |
| `clean` | Remove build artifacts |
| `all` | Full workflow: generate → clean → compile → run → view |

## Neural Network Top Module

The `neural_network.vhd` module provides a complete neural network with configurable layers and automated weight loading.

### Architecture

**Generics:**
- `num_inputs`: Number of network inputs (e.g., 4)
- `num_layers`: Number of layers, 1-4 (e.g., 3 for input→hidden→output)
- `layer_size_0..3`: Neurons per layer
- `data_width`: Bit width (default: 32)
- `use_sigmoid`: Activation function selector

**State Machine:**
```
IDLE → LOADING → READY → INFERENCE → OUTPUT → READY
```

**Example Configuration:**
- Network: 4 inputs → 4 neurons → 3 neurons → 2 outputs
- Generics: `num_inputs=4, num_layers=3, layer_size_0=4, layer_size_1=3, layer_size_2=2`

### Quick Test

```bash
# Generate neural network testbench
make generate-nn-tb

# Run simulation
make TESTBENCH=neural_network

# Or use custom model
.venv/bin/python scripts/gen_neural_network.py \
  --model-file my_model.h5 \
  --num-tests 10 \
  --output testbench/my_nn_tb.vhd
```

## Neural Network Weight Loading

### Hardware Interface

Each neuron has a weight loading interface:
- `load_enable`: Enable weight loading
- `weight_data_i`: Weight/bias value (fixed-point)
- `weight_index_i`: Index (0..num_inputs-1 for weights, num_inputs for bias)

Layers add neuron selection:
- `neuron_select`: Which neuron in the layer to load

### Loading Sequence

1. Set `load_enable = '1'`
2. Set `neuron_select` (for layers)
3. Set `weight_index_i`
4. Set `weight_data_i`
5. Clock edge → weight stored
6. Repeat for all weights/biases
7. Set `load_enable = '0'` for inference

### Using Keras Models

```python
# Train a model
model = keras.Sequential([
    keras.layers.Dense(4, activation='sigmoid', input_dim=4),
    keras.layers.Dense(3, activation='sigmoid'),
    keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)

# Generate testbench
gen_neural_network.py --model-file model.h5 --output testbench/nn_tb.vhd
```

## Configuration

Edit `scripts/config.py` to modify:
- LUT size (default: 256 entries)
- Data width (default: 32 bits)
- Fractional width (default: 16 bits)
- Input range for LUT (default: -8.0 to 8.0)

## Testing

**Component Tests:**
- ✓ `activation_func_tb`: 16/16 PASS
- ✓ `neuron_tb`: 5/5 PASS
- ✓ `layer_tb`: 3/3 PASS (with external weights)

**Neural Network Tests:**
- ✓ `neural_network_tb`: 10/10 PASS (5 test cases × 2 outputs)
  - Network: 4 → 4 → 3 → 2
  - Tolerance: 0.046875
  - Hardware matches Keras model within tolerance

## Development

### Adding New Components

1. Write VHDL in `src/`
2. Create testbench in `testbench/`
3. Add to `Makefile` FILES list
4. Run `make TESTBENCH=your_component`

### Modifying Code Generation

Edit scripts in `scripts/`:
- `gen_sigmoid_pkg.py` - LUT and package generation
- `gen_testbenches.py` - Component testbenches
- `gen_neural_network.py` - Neural network testbenches

## License

Educational project - use freely.

## References

- VHDL IEEE 1076-2008
- IEEE fixed-point package
- TensorFlow/Keras documentation

## Unreleased

### HDL Bug Fixes
- control_unit: Fix TRAIN_FWD/BWD FSM sync (calc_done handshake)
- memory_control_unit: Fix read-modify-write race and lost read requests during updates
- calculation_unit: Support multi-output networks (output_data is now bus array)
- calculation_unit: Latch mode at computation start to prevent mid-run corruption
- calculation_unit: Per-layer activation config via USE_SIGMOID boolean_array generic
- activation_func: Add proper ReLU architecture alongside sigmoid

### Structure Cleanup
- Move types.vhd, sigmoid_lut_pkg.vhd, pkg_layer.vhd to src/packages/
- Update sources.mk with correct paths
- Create data/ directory for generated weight/hex files
- Remove broken neural_network_tb.vhd (incompatible with current interface)

### Python Tool Unification
- Add scripts/configure.py as single entry point for all code generation
- Generate types.vhd from YAML config
- Add 'make configure' target
- Remove deprecated scripts/config.py

### ONNX Integration
- Add scripts/onnx_parser.py for parsing ONNX model files
- Integrate --model flag into configure.py
- Export weights to hex files in data/
- Add onnx to requirements.txt

### sfixed Port Conversion
- Convert all layer entity ports from std_logic_vector to sfixed_bus/sfixed_bus_array
- Affected files: conv2d_layer, maxpool_layer, avgpool_layer, flatten_layer, rnn_cell, lstm_cell, rnn_layer
- Fix fixed-point constant initialization in lstm_cell and avgpool_layer
- Update XOR testbenches for sfixed output ports

### cocotb Test Framework
- Add cocotb-based Python testbenches replacing VHDL testbench generators
- VHDL wrappers with flattened std_logic_vector ports (GHDL VPI workaround)
- Tests for activation_func, neuron, layer, and full XOR network inference
- Add cocotb-test Makefile target and pytest runner

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

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>

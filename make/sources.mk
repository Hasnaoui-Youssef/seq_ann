FILES = src/packages/types.vhd \
        src/packages/pkg_layer.vhd \
        src/packages/sigmoid_lut_pkg.vhd \
        src/packages/layer_interface_pkg.vhd \
        src/core/accumulator/half_adder.vhd \
        src/core/accumulator/signed_mult.vhd \
        src/core/neuron/activation_func.vhd \
        src/core/accumulator/full_adder.vhd \
        src/core/accumulator/n_bit_adder.vhd \
        src/core/accumulator/acc.vhd \
        src/core/layer/weight_bank.vhd \
        src/core/neuron/neuron.vhd \
        src/core/layer/layer.vhd \
        src/layers/conv2d_layer.vhd \
        src/layers/maxpool_layer.vhd \
        src/layers/avgpool_layer.vhd \
        src/layers/flatten_layer.vhd \
        src/layers/rnn_cell.vhd \
        src/layers/lstm_cell.vhd \
        src/layers/rnn_layer.vhd \
        src/memory/bram.vhd \
        src/units/memory_control_unit.vhd \
        src/units/calculation_unit.vhd \
        src/units/control_unit.vhd \
        src/neural_network.vhd

# GHDL Flags
GHDL_FLAGS = --std=08 --ieee=synopsys -frelaxed --workdir=work

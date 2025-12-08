# Source files (sorted by dependency)
FILES = src/types.vhd \
        src/pkg_layer.vhd \
        src/sigmoid_lut_pkg.vhd \
        src/half_adder.vhd \
        src/signed_mult.vhd \
        src/activation_func.vhd \
        src/full_adder.vhd \
        src/n_bit_adder.vhd \
        src/acc.vhd \
        src/weight_bank.vhd \
        src/neuron.vhd \
        src/layer.vhd \
        src/bram.vhd \
        src/memory_control_unit.vhd \
        src/calculation_unit.vhd \
        src/control_unit.vhd \
        src/neural_network.vhd

# GHDL Flags
GHDL_FLAGS = --std=08 --ieee=synopsys -frelaxed --workdir=work

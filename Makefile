# vhdl files
FILES =	src/types.vhd 					\
		src/sigmoid_lut_pkg.vhd 		\
		src/half_adder.vhd 				\
		src/full_adder.vhd 				\
		src/n_bit_adder.vhd 			\
		src/activation_func.vhd 		\
		src/acc.vhd 					\
		src/neuron.vhd					\
		src/layer.vhd					\
		src/neural_network.vhd			#\


# testbench
TESTBENCHPATH = testbench/${TESTBENCHFILE}.vhd
TESTBENCHFILE = ${TESTBENCH}_tb
WORKDIR = work
XMLDIR = xml

#GHDL CONFIG
GHDL_CMD = ghdl
GHDL_FLAGS  = --std=08 --ieee=synopsys --warn-no-vital-generic --workdir=$(WORKDIR)

STOP_TIME = 1000ns
# Simulation break condition
#GHDL_SIM_OPT = --assert-level=error
GHDL_SIM_OPT = --stop-time=$(STOP_TIME)

# WAVEFORM_VIEWER = flatpak run io.github.gtkwave.GTKWave
WAVEFORM_VIEWER = gtkwave

.PHONY: clean generate generate-pkg generate-tb test-all

# Code generation targets
generate: generate-pkg generate-tb

generate-pkg:
	@echo "Generating sigmoid LUT package..."
	@python3 scripts/gen_sigmoid_pkg.py

generate-tb:
	@echo "Generating testbenches..."
	@python3 scripts/gen_testbenches.py

generate-nn-tb:
	@echo "Generating neural network testbench..."
	@.venv/bin/python scripts/gen_neural_network.py --create-model --layers 4,3,2 --num-tests 5

# Run all testbenches
test-all: generate
	@echo "========================================="
	@echo "Running all testbenches..."
	@echo "========================================="
	@echo "\n[1/3] Testing activation_func..."
	@$(MAKE) TESTBENCH=activation_func make run || echo "activation_func test FAILED"
	@echo "\n[2/3] Testing neuron..."
	@$(MAKE) TESTBENCH=neuron make run || echo "neuron test FAILED"
	@echo "\n[3/3] Testing layer..."
	@$(MAKE) TESTBENCH=layer make run || echo "layer test FAILED"
	@echo "\n========================================="
	@echo "All tests completed!"
	@echo "========================================="

all: generate clean make run view

xml:
	@echo "Generating XML info"
	@$(GHDL_CMD) --file-to-xml $(GHDL_FLAGS) $(FILES)

make:
ifeq ($(strip $(TESTBENCH)),)
	@echo "TESTBENCH not set. Use TESTBENCH=<value> to set it."
	@exit 1
endif

	@mkdir -p $(WORKDIR)
	@$(GHDL_CMD) -a $(GHDL_FLAGS) $(FILES)
	@$(GHDL_CMD) -a $(GHDL_FLAGS) $(TESTBENCHPATH)
	@$(GHDL_CMD) -e $(GHDL_FLAGS) $(TESTBENCHFILE)

run:
	@$(GHDL_CMD) -r $(GHDL_FLAGS) --workdir=$(WORKDIR) $(TESTBENCHFILE) --wave=$(TESTBENCHFILE).ghw $(GHDL_SIM_OPT)
	@mv $(TESTBENCHFILE).ghw $(WORKDIR)/

view:
	@$(WAVEFORM_VIEWER) --dump=$(WORKDIR)/$(TESTBENCHFILE).ghw

clean:
	@rm -rf $(WORKDIR)

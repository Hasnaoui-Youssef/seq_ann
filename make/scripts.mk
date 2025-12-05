# Python Scripts
PYTHON = .venv/bin/python

# Python script parameters
LUT_SIZE ?= 256
DATA_WIDTH ?= 32
FRAC_BITS ?= 16
NUM_TESTS ?= 10
LAYERS ?= 4,3,2

.PHONY: generate generate-pkg generate-tb generate-nn-tb generate-xor-tb

generate: generate-pkg generate-tb

generate-pkg:
	@echo "Generating sigmoid LUT package..."
	@$(PYTHON) scripts/gen_sigmoid_pkg.py \
		--lut-size $(LUT_SIZE) \
		--data-width $(DATA_WIDTH) \
		--frac-bits $(FRAC_BITS)

generate-tb:
	@echo "Generating testbenches..."
	@$(PYTHON) scripts/gen_testbenches.py \
		--data-width $(DATA_WIDTH) \
		--frac-bits $(FRAC_BITS) \
		--num-tests $(NUM_TESTS)

generate-nn-tb:
	@echo "Generating neural network testbench..."
	@$(PYTHON) scripts/gen_neural_network.py \
		--create-model \
		--layers $(LAYERS) \
		--num-tests $(NUM_TESTS) \
		--data-width $(DATA_WIDTH) \
		--frac-bits $(FRAC_BITS)

generate-xor-tb:
	@echo "Generating XOR testbench..."
	@$(PYTHON) scripts/gen_neural_network.py \
		--xor \
		--output testbench/xor_tb.vhd \
		--data-width $(DATA_WIDTH) \
		--frac-bits $(FRAC_BITS)

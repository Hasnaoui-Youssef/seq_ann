# Python Scripts
PYTHON = .venv/bin/python

# Configuration file (searches for *.nn_conf.yaml in root, falls back to configs/default.nn_conf.yaml)
CONFIG ?=

# Python script parameters (legacy, kept for backward compatibility with gen_testbenches.py)
NUM_TESTS ?= 10
LAYERS ?= 4,3,2

# Config argument helper
ifdef CONFIG
CONFIG_ARG = --config $(CONFIG)
else
CONFIG_ARG =
endif

.PHONY: generate generate-pkg generate-tb generate-nn-tb generate-xor-tb

generate: generate-pkg generate-tb

generate-pkg:
	@echo "Generating sigmoid LUT package..."
	@$(PYTHON) scripts/gen_sigmoid_pkg.py $(CONFIG_ARG)

generate-tb:
	@echo "Generating testbenches..."
	@$(PYTHON) scripts/gen_testbenches.py $(CONFIG_ARG) \
		--num-tests $(NUM_TESTS)

generate-nn-tb:
	@echo "Generating neural network testbench..."
	@$(PYTHON) scripts/gen_neural_network.py $(CONFIG_ARG) \
		--create-model \
		--layers $(LAYERS) \
		--num-tests $(NUM_TESTS)

generate-xor-tb:
	@echo "Generating XOR testbench..."
	@$(PYTHON) scripts/gen_neural_network.py $(CONFIG_ARG) \
		--xor \
		--output testbench/xor_tb.vhd

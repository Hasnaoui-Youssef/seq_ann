# Main Makefile

include make/sources.mk
include make/scripts.mk
include make/testbenches.mk

.PHONY: clean test test-all cocotb-test

# Default: complete workflow with waveform viewing
all: generate clean compile run view

# Headless test (no waveform viewer)
test: generate clean compile run

# Headless test all components
test-all:
	@$(MAKE) TESTBENCH=activation_func test
	@$(MAKE) TESTBENCH=neuron test
	@$(MAKE) TESTBENCH=layer test
	@$(MAKE) TESTBENCH=xor test

# cocotb tests via pytest
cocotb-test: generate
	@echo "Running cocotb tests..."
	@cd $(CURDIR) && python -m pytest tests/run.py -v $(PYTEST_ARGS)

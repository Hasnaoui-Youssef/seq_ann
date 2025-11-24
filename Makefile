# Main Makefile

include make/sources.mk
include make/scripts.mk
include make/testbenches.mk

.PHONY: clean

# Default: complete workflow with waveform viewing
all: generate clean compile run view

# Test all components (without opening viewer for each)
test-all:
	@echo "Running all tests..."
	@make TESTBENCH=activation_func generate clean compile run
	@make TESTBENCH=neuron generate clean compile run
	@make TESTBENCH=layer generate clean compile run
	@make TESTBENCH=xor generate clean compile run
	@echo "All tests complete. Waveforms available in $(WORKDIR)/"

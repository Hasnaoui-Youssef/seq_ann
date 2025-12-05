# Main Makefile

include make/sources.mk
include make/scripts.mk
include make/testbenches.mk

.PHONY: clean

# Default: complete workflow with waveform viewing
all: generate clean compile run view

# Headless test (no GTKWave)
test: generate clean compile run

# Test all components (without opening viewer for each)
test-all:
	@echo "Running all tests..."
	@make TESTBENCH=activation_func test
	@make TESTBENCH=neuron test
	@make TESTBENCH=layer test
	@make TESTBENCH=xor test
	@echo "All tests complete. Waveforms available in $(WORKDIR)/"

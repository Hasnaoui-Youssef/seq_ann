# Main Makefile

include make/sources.mk
include make/scripts.mk

.PHONY: clean test cocotb-test configure

# Run all cocotb tests
test: configure cocotb-test

# cocotb tests via pytest
cocotb-test: configure
@echo "Running cocotb tests..."
@cd $(CURDIR) && python -m pytest tests/run.py -v $(PYTEST_ARGS)

# Clean build artifacts
clean:
@echo "Cleaning..."
@rm -rf work/ sim_build/ *.cf results.xml

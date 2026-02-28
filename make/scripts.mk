# Python Scripts
PYTHON = .venv/bin/python

# Configuration file (searches for *.nn_conf.yaml in root, falls back to configs/default.nn_conf.yaml)
CONFIG ?=

# Python script parameters
NUM_TESTS ?= 10

# Config argument helper
ifdef CONFIG
CONFIG_ARG = --config $(CONFIG)
else
CONFIG_ARG =
endif

.PHONY: configure generate generate-pkg generate-tb

# Unified configure: generates all VHDL packages and testbenches from YAML config
configure:
	@$(PYTHON) scripts/configure.py $(CONFIG_ARG) --num-tests $(NUM_TESTS)

# Legacy targets (call unified configure or individual scripts)
generate: configure

generate-pkg:
	@echo "Generating sigmoid LUT package..."
	@$(PYTHON) scripts/gen_sigmoid_pkg.py $(CONFIG_ARG)

generate-tb:
	@echo "Generating testbenches..."
	@$(PYTHON) scripts/gen_testbenches.py $(CONFIG_ARG) \
		--num-tests $(NUM_TESTS)

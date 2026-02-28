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

.PHONY: configure generate generate-pkg generate-tb mnist-harness timeseries-harness

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

# MNIST test harness
MODEL_TYPE ?= dense
TRAIN_EPOCHS ?= 5
NUM_TEST_SAMPLES ?= 10

mnist-harness:
	@echo "Running MNIST test harness ($(MODEL_TYPE))..."
	@$(PYTHON) scripts/mnist_harness.py \
		--model-type $(MODEL_TYPE) \
		--epochs $(TRAIN_EPOCHS) \
		--num-test-samples $(NUM_TEST_SAMPLES)

# Timeseries test harness
RNN_TYPE ?= rnn
SEQ_LEN ?= 10
HIDDEN_SIZE ?= 16

timeseries-harness:
	@echo "Running timeseries test harness ($(RNN_TYPE))..."
	@$(PYTHON) scripts/timeseries_harness.py \
		--model-type $(RNN_TYPE) \
		--epochs $(TRAIN_EPOCHS) \
		--seq-len $(SEQ_LEN) \
		--hidden-size $(HIDDEN_SIZE) \
		--num-test-samples $(NUM_TEST_SAMPLES)

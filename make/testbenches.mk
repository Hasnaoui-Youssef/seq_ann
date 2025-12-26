# Testbench targets

# Configurable variables
WORKDIR ?= work
TESTBENCH ?= activation_func
TESTBENCHFILE ?= testbench/$(TESTBENCH)_tb.vhd

# Simulation options
SIM_STOP_TIME ?= 100us
SIM_ASSERT_LEVEL ?= error
WAVE_FORMAT ?= ghw
WAVEFILE = $(WORKDIR)/$(TESTBENCH).$(WAVE_FORMAT)

# GHDL flags
GHDL_FLAGS = --workdir=$(WORKDIR) --std=08 --ieee=synopsys -frelaxed --warn-no-vital-generic

# Compilation
compile: $(FILES) $(TESTBENCHFILE)
	@mkdir -p $(WORKDIR)
	@echo "Compiling sources..."
	@ghdl -a $(GHDL_FLAGS) $(FILES)
	@echo "Compiling testbench..."
	@ghdl -a $(GHDL_FLAGS) $(TESTBENCHFILE)
	@echo "Elaborating..."
	@ghdl -e $(GHDL_FLAGS) $(TESTBENCH)_tb

# Execution
run:
	@echo "Running simulation..."
	@ghdl -r $(GHDL_FLAGS) $(TESTBENCH)_tb \
		--wave=$(WAVEFILE) \
		--stop-time=$(SIM_STOP_TIME) \
		--assert-level=$(SIM_ASSERT_LEVEL)

# Viewer
view:
	@echo "Opening waveform..."
	@gtkwave $(WAVEFILE)

# Clean
clean:
	@echo "Cleaning..."
	@rm -rf $(WORKDIR) *.cf

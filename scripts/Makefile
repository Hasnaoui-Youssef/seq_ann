# vhdl files
FILES =	src/types.vhd 		\
		src/sigmoid_lut_pkg.vhd 		\
		src/half_adder.vhd 	\
		src/full_adder.vhd 	\
		src/n_bit_adder.vhd \
		src/acc.vhd \


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

.PHONY: clean

all: clean make run view

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

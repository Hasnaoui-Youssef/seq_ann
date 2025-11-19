#!/usr/bin/env python3

import math
import argparse
from pathlib import Path


class SigmoidConfig:
    def __init__(self, lut_size=256, input_width=48, output_width=32, num_test_inputs=16, input_range=(-8.0, 8.0)):
        self.lut_size = lut_size
        self.input_width = input_width
        self.output_width = output_width
        self.num_test_inputs = num_test_inputs
        self.input_range = input_range
        self.lut_bits = (lut_size - 1).bit_length()
        self.data_width = input_width // 2


    def get_index_range(self):
        if self.lut_bits <= self.data_width:
            high_bit = (self.lut_bits // 2) - 1
            low_bit = - (self.lut_bits - (self.lut_bits // 2))
            return (high_bit, low_bit)

        high_bit = (self.data_width // 2) - 1
        low_bit = - (self.data_width // 2)
        return (high_bit, low_bit)


def sigmoid(x):
    try:
        return 1.0/ (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def generate_sigmoid_lut(config : SigmoidConfig):
    lut = []
    x_min, x_max = config.input_range
    for i in range(config.lut_size):
        x = x_min + (x_max - x_min) * i / (config.lut_size - 1)
        sig_val = sigmoid(x)
        lut.append((i, x, sig_val))

    return lut

def generate_vhdl_lut_package(config : SigmoidConfig, lut, output_file="src/sigmoid_lut_pkg.vhd"):
    high_bit, low_bit = config.get_index_range()
    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
    package sigmoid_lut_pkg is
        constant LUT_SIZE : integer := {config.lut_size};
        constant LUT_BITS : integer := {config.lut_bits};
        constant INPUT_WIDTH : integer := {config.input_width};
        constant OUTPUT_WIDTH : integer := {config.output_width};
        constant INDEX_HIGH : integer := {high_bit};
        constant INDEX_LOW : integer := {low_bit};
        constant INDEX_WIDTH : integer := INDEX_HIGH - INDEX_LOW + 1;
        constant INPUT_MAX : real := {config.input_range[1]};
        constant INPUT_MIN : real := {config.input_range[0]};

        type sigmoid_lut_type is array(0 to LUT_SIZE - 1) of real;

        constant SIGMOID_LUT : sigmoid_lut_type := (
"""
    for i, (idx, x, sig_val) in enumerate(lut):
        if i == len(lut) - 1:
            vhdl_code += f"        {i} => {sig_val:.10f}\n"
        else:
            vhdl_code += f"        {i} => {sig_val:.10f},\n"

    vhdl_code += """        );

end package sigmoid_lut_pkg;
    """

    Path(output_file).parent.mkdir(parents = True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print("Generated LUT package file")
    return output_file


def generate_activation_func_vhdl(config : SigmoidConfig, output_file="src/activation_func.vhd"):
    """Generate activation function VHDL with sigmoid LUT"""
    high_bit, low_bit = config.get_index_range()
    needs_padding = config.lut_bits > config.data_width

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func is
    generic(
        input_width : integer := {config.input_width};
        output_width : integer := {config.output_width}
    );
    port(
        input_i : in std_logic_vector(input_width - 1 downto 0);
        output_o : out sfixed(output_width / 2 - 1 downto - (output_width / 2)) := (others => '0')
    );
end entity activation_func;

architecture relu of activation_func is
begin
    output_o <= to_sfixed(input_i, output_o) when signed(input_i) >= 0 else to_sfixed_a(0);
end architecture relu;

architecture sigmoid of activation_func is
    signal input_sfixed : sfixed((input_width + 1) / 2 - 1 downto - (input_width / 2));
"""

    if needs_padding:
        vhdl_code += """    signal index_bits_padded : sfixed((LUT_BITS + 1)/2 - 1 downto -(LUT_BITS/2));
"""

    vhdl_code += """    signal lut_index : integer range 0 to LUT_SIZE - 1;
    signal sigmoid_value : real;
    signal input_real : real;
    signal clipped : real;
    signal normalized : real;

begin
    -- Convert input to sfixed
    input_sfixed <= to_sfixed(input_i, input_sfixed);
    input_real <= to_real(input_sfixed);
    clipped <= INPUT_MAX when input_real > INPUT_MAX else INPUT_MIN when input_real < INPUT_MIN else input_real;
    normalized <= ((clipped - INPUT_MIN)/(INPUT_MAX - INPUT_MIN));

"""

    vhdl_code += f"""    -- Extract middle bits for LUT indexing ({config.lut_bits} bits from input)
    lut_index <= 0 when (normalized  < 0.0)
                 else (LUT_SIZE - 1) when (normalized > 1.0)
                 else integer(normalized * real(LUT_SIZE - 1));

"""

    vhdl_code += """
    -- Lookup sigmoid value from LUT
    sigmoid_value <= SIGMOID_LUT(lut_index);

    -- Convert to output fixed-point format
    output_o <= to_sfixed(sigmoid_value, output_o'high, output_o'low);

end architecture sigmoid;
"""

    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated activation function VHDL: {output_file}")
    return output_file

def generate_activation_func_testbench(config, lut, tolerance, output_file="testbench/activation_func_tb.vhd"):
    """Generate VHDL testbench for sigmoid activation function"""

    # Generate test vectors
    test_vectors = []
    x_min, x_max = config.input_range

    for i in range(config.num_test_inputs):
        x = x_min + (x_max - x_min) * i / (config.num_test_inputs - 1)
        expected = sigmoid(x)
        test_vectors.append((x, expected))

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func_tb is
end entity activation_func_tb;

architecture testbench of activation_func_tb is
    -- Configuration constants
    constant INPUT_WIDTH_C : integer := {config.input_width};
    constant OUTPUT_WIDTH_C : integer := {config.output_width};

    -- Component declaration
    component activation_func is
        generic(
            input_width : integer := INPUT_WIDTH_C;
            output_width : integer := OUTPUT_WIDTH_C
        );
        port(
            input_i : in std_logic_vector(input_width / 2 - 1 downto -(input_width / 2));
            output_o : out sfixed(output_width / 2 - 1 downto -(output_width / 2))
        );
    end component;

    -- Test signals (std_logic_vector uses natural range)
    signal input_s : std_logic_vector(INPUT_WIDTH_C - 1 downto 0);
    signal output_s : sfixed(OUTPUT_WIDTH_C / 2 - 1 downto -(OUTPUT_WIDTH_C / 2));

    -- Helper signals
    signal input_sfixed : sfixed(INPUT_WIDTH_C / 2 - 1 downto -(INPUT_WIDTH_C / 2));
    signal input_real : real;
    signal output_real : real;

    -- Test configuration
    constant NUM_TESTS : integer := {config.num_test_inputs};
    type real_array is array (0 to NUM_TESTS - 1) of real;

    -- Test vectors: input values
    constant TEST_INPUTS : real_array := (
"""

    for i, (x, _) in enumerate(test_vectors):
        if i == len(test_vectors) - 1:
            vhdl_code += f"        {i} => {x:.6f}\n"
        else:
            vhdl_code += f"        {i} => {x:.6f},\n"

    vhdl_code += """    );

    -- Expected outputs
    constant EXPECTED_OUTPUTS : real_array := (
"""

    for i, (_, expected) in enumerate(test_vectors):
        if i == len(test_vectors) - 1:
            vhdl_code += f"        {i} => {expected:.10f}\n"
        else:
            vhdl_code += f"        {i} => {expected:.10f},\n"

    vhdl_code += f"""    );

    constant TOLERANCE : real := {tolerance};  -- {tolerance * 100}% tolerance for comparison

begin
    -- DUT instantiation (sigmoid architecture)
    dut: entity work.activation_func(sigmoid)
        generic map(
            input_width => INPUT_WIDTH_C,
            output_width => OUTPUT_WIDTH_C
        )
        port map(
            input_i => input_s,
            output_o => output_s
        );

    -- Convert signals for monitoring
    input_sfixed <= to_sfixed(input_s, input_sfixed'high, input_sfixed'low);
    input_real <= to_real(input_sfixed);
    output_real <= to_real(output_s);

    -- Test process
    test_proc: process
        variable error : real;
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
    begin
        report "========================================";
        report "Starting Sigmoid Activation Function Test";
        report "LUT Size: " & integer'image(LUT_SIZE);
        report "Input Width: " & integer'image(INPUT_WIDTH_C);
        report "Output Width: " & integer'image(OUTPUT_WIDTH_C);
        report "========================================";

        -- Run tests
        for i in 0 to NUM_TESTS - 1 loop
            -- Set input
            input_s <= to_slv(to_sfixed(TEST_INPUTS(i), input_sfixed'high, input_sfixed'low));

            -- Wait for computation
            wait for 10 ns;

            -- Check output
            error := abs(output_real - EXPECTED_OUTPUTS(i));

            report "Test " & integer'image(i) & ": " &
                   "Input = " & real'image(input_real) &
                   ", Output = " & real'image(output_real) &
                   ", Expected = " & real'image(EXPECTED_OUTPUTS(i)) &
                   ", Error = " & real'image(error);

            if error < TOLERANCE then
                pass_count := pass_count + 1;
                report "  PASS";
            else
                fail_count := fail_count + 1;
                report "  FAIL - Error exceeds tolerance!";
            end if;

            wait for 10 ns;
        end loop;

        -- Summary
        report "========================================";
        report "Test Summary:";
        report "  Passed: " & integer'image(pass_count) & "/" & integer'image(NUM_TESTS);
        report "  Failed: " & integer'image(fail_count) & "/" & integer'image(NUM_TESTS);
        report "========================================";

        if fail_count = 0 then
            report "ALL TESTS PASSED!" severity note;
        else
            report "SOME TESTS FAILED!" severity warning;
        end if;

        wait;
    end process test_proc;

end architecture testbench;
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated testbench: {output_file}")
    return output_file

# ============================================================================
# Makefile Update
# ============================================================================

def update_makefile(makefile_path="Makefile"):
    """Update Makefile to include sigmoid LUT package"""

    with open(makefile_path, 'r') as f:
        content = f.read()

    # Check if sigmoid_lut_pkg is already in FILES
    if 'sigmoid_lut_pkg.vhd' not in content:
        # Add after types.vhd
        new_content = content.replace(
            'FILES =\tsrc/types.vhd',
            'FILES =\tsrc/types.vhd \t\t\\\n\t\tsrc/sigmoid_lut_pkg.vhd'
        )

        with open(makefile_path, 'w') as f:
            f.write(new_content)

        print(f"Updated {makefile_path} to include sigmoid_lut_pkg.vhd")
    else:
        print(f"{makefile_path} already includes sigmoid_lut_pkg.vhd")

# ============================================================================
# Main Workflow
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate sigmoid LUT and testbench for VHDL activation function'
    )
    parser.add_argument('--lut-size', type=int, default=256,
                        help='Number of LUT entries (default: 256)')
    parser.add_argument('--input-width', type=int, default=48,
                        help='Input width in bits (default: 48)')
    parser.add_argument('--output-width', type=int, default=32,
                        help='Output width in bits (default: 32)')
    parser.add_argument('--num-tests', type=int, default=16,
                        help='Number of test vectors (default: 16)')
    parser.add_argument('--input-min', type=float, default=-8.0,
                        help='Minimum input value (default: -8.0)')
    parser.add_argument('--input-max', type=float, default=8.0,
                        help='Maximum input value (default: 8.0)')
    parser.add_argument('--test-tolerance', type=float, default=0.1,
                        help='Accuracy of test results (default : 0.1)')

    args = parser.parse_args()

    test_tolerance = args.test_tolerance

    # Create configuration
    config = SigmoidConfig(
        lut_size=args.lut_size,
        input_width=args.input_width,
        output_width=args.output_width,
        num_test_inputs=args.num_tests,
        input_range=(args.input_min, args.input_max)
    )

    print("=" * 60)
    print("Sigmoid Activation Function Workflow")
    print("=" * 60)
    print(f"LUT Size: {config.lut_size} entries ({config.lut_bits} bits)")
    print(f"Input Width: {config.input_width} bits")
    print(f"Output Width: {config.output_width} bits")
    print(f"Data Width: {config.data_width} bits")

    high_bit, low_bit = config.get_index_range()
    print(f"Index Range: input_sfixed({high_bit} downto {low_bit})")
    print(f"Input Range: [{config.input_range[0]}, {config.input_range[1]}]")
    print(f"Number of Tests: {config.num_test_inputs}")
    print("=" * 60)

    # Generate LUT
    print("\n[1/4] Generating sigmoid LUT...")
    lut = generate_sigmoid_lut(config)

    # Generate VHDL LUT package
    print("\n[2/4] Generating VHDL LUT package...")
    generate_vhdl_lut_package(config, lut)

    # Generate activation function VHDL
    print("\n[3/4] Generating activation function VHDL...")
    generate_activation_func_vhdl(config)

    # Generate testbench
    print("\n[4/4] Generating testbench...")
    generate_activation_func_testbench(config, lut, test_tolerance)

    # Update Makefile
    print("\n[5/4] Updating Makefile...")
    update_makefile()

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print("\nTo run the simulation:")
    print("  make TESTBENCH=activation_func")
    print("\nGenerated files:")
    print("  - src/sigmoid_lut_pkg.vhd")
    print("  - src/activation_func.vhd")
    print("  - testbench/activation_func_tb.vhd")
    print("=" * 60)

if __name__ == "__main__":
    main()


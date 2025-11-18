library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func_tb is
end entity activation_func_tb;

architecture testbench of activation_func_tb is
    -- Configuration constants
    constant INPUT_WIDTH_C : integer := 48;
    constant OUTPUT_WIDTH_C : integer := 32;

    -- Component declaration

    -- Test signals (std_logic_vector uses natural range)
    signal input_s : std_logic_vector(INPUT_WIDTH_C - 1 downto 0);
    signal output_s : sfixed(OUTPUT_WIDTH_C / 2 - 1 downto -(OUTPUT_WIDTH_C / 2));

    -- Helper signals
    signal input_sfixed : sfixed(INPUT_WIDTH_C / 2 - 1 downto -(INPUT_WIDTH_C / 2));
    signal input_real : real;
    signal output_real : real;

    -- Test configuration
    constant NUM_TESTS : integer := 16;
    type real_array is array (0 to NUM_TESTS - 1) of real;

    -- Test vectors: input values
    constant TEST_INPUTS : real_array := (
        0 => -8.000000,
        1 => -6.933333,
        2 => -5.866667,
        3 => -4.800000,
        4 => -3.733333,
        5 => -2.666667,
        6 => -1.600000,
        7 => -0.533333,
        8 => 0.533333,
        9 => 1.600000,
        10 => 2.666667,
        11 => 3.733333,
        12 => 4.800000,
        13 => 5.866667,
        14 => 6.933333,
        15 => 8.000000
    );

    -- Expected outputs
    constant EXPECTED_OUTPUTS : real_array := (
        0 => 0.0003353501,
        1 => 0.0009737971,
        2 => 0.0028242994,
        3 => 0.0081625712,
        4 => 0.0233545165,
        5 => 0.0649691691,
        6 => 0.1679816149,
        7 => 0.3697397771,
        8 => 0.6302602229,
        9 => 0.8320183851,
        10 => 0.9350308309,
        11 => 0.9766454835,
        12 => 0.9918374288,
        13 => 0.9971757006,
        14 => 0.9990262029,
        15 => 0.9996646499
    );

    constant TOLERANCE : real := 0.01;  -- 1% tolerance for comparison

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

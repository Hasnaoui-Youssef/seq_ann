library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func_tb is
end entity activation_func_tb;

architecture testbench of activation_func_tb is
    -- Test signals
    signal input_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal output_s : sfixed(INT_BITS - 1 downto -FRAC_BITS);

    -- Helper signals
    signal input_sfixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
    signal input_real : real;
    signal output_real : real;

    -- Test configuration
    constant NUM_TESTS : integer := 10;
    type real_array is array (0 to NUM_TESTS - 1) of real;

    -- Test vectors: input values
    constant TEST_INPUTS : real_array := (
        0 => -8.000000,
        1 => -6.222222,
        2 => -4.444444,
        3 => -2.666667,
        4 => -0.888889,
        5 => 0.888889,
        6 => 2.666667,
        7 => 4.444444,
        8 => 6.222222,
        9 => 8.000000
    );

    -- Expected outputs
    constant EXPECTED_OUTPUTS : real_array := (
        0 => 0.0003353501,
        1 => 0.0019808978,
        2 => 0.0116073164,
        3 => 0.0649691691,
        4 => 0.2913391750,
        5 => 0.7086608250,
        6 => 0.9350308309,
        7 => 0.9883926836,
        8 => 0.9980191022,
        9 => 0.9996646499
    );

    constant TOLERANCE : real := 0.01;  -- 1% tolerance for comparison

begin
    -- DUT instantiation (sigmoid architecture)
    dut: entity work.activation_func(sigmoid)
        generic map(
            input_width => DATA_WIDTH
        )
        port map(
            input_i => input_s,
            output_o => output_s
        );

    -- Convert signals for monitoring
    input_sfixed <= to_sfixed(input_s, input_sfixed);
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
        report "Input Width: " & integer'image(DATA_WIDTH);
        report "Output Width: " & integer'image(DATA_WIDTH);
        report "========================================";

        -- Run tests
        for i in 0 to NUM_TESTS - 1 loop
            -- Set input
            input_s <= to_slv(to_sfixed(TEST_INPUTS(i), input_sfixed));

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

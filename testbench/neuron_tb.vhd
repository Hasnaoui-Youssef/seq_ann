library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.math_real.all;

use work.types.all;

entity neuron_tb is
end entity neuron_tb;

architecture testbench of neuron_tb is
    constant NUM_INPUTS_C : integer := 4;
    constant DATA_WIDTH_C : integer := 16;
    constant USE_SIGMOID_C : boolean := true;

    signal inputs_s : std_logic_bus_array(NUM_INPUTS_C - 1 downto 0)(DATA_WIDTH_C - 1 downto 0);
    signal weights_s : std_logic_bus_array(NUM_INPUTS_C downto 0)(DATA_WIDTH_C - 1 downto 0);
    signal output_s : std_logic_vector(DATA_WIDTH_C - 1 downto 0);
    signal overflow_s : std_logic;

    -- Helper signals for monitoring
    type real_array is array (integer range <>) of real;
    signal inputs_real : real_array(NUM_INPUTS_C - 1 downto 0);
    signal weights_real : real_array(NUM_INPUTS_C downto 0);
    signal output_real : real;

    -- Test vectors
    type test_case is record
        inputs : real_array(NUM_INPUTS_C - 1 downto 0);
        weights : real_array(NUM_INPUTS_C downto 0);  -- Last is bias
        expected_sum : real;
        description : string;
    end record;

    type test_case_array is array (natural range <>) of test_case;

    constant TEST_CASES : test_case_array := (
        -- Test 1: Simple positive values
        (
            inputs => (0.5, 0.5, 0.5, 0.5),
            weights => (1.0, 1.0, 1.0, 1.0, 0.0),  -- bias = 0
            expected_sum => 2.0,  -- 0.5*1 + 0.5*1 + 0.5*1 + 0.5*1 = 2.0
            description => "Simple positive values              "
        ),
        -- Test 2: With bias
        (
            inputs => (1.0, 1.0, 1.0, 1.0),
            weights => (0.5, 0.5, 0.5, 0.5, 1.0),  -- bias = 1.0
            expected_sum => 3.0,  -- 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 + 1.0 = 3.0
            description => "With positive bias                  "
        ),
        -- Test 3: Negative values
        (
            inputs => (-1.0, -1.0, 1.0, 1.0),
            weights => (1.0, 1.0, 1.0, 1.0, 0.0),
            expected_sum => 0.0,  -- -1*1 + -1*1 + 1*1 + 1*1 = 0
            description => "Mixed positive and negative         "
        ),
        -- Test 4: All zeros
        (
            inputs => (0.0, 0.0, 0.0, 0.0),
            weights => (1.0, 1.0, 1.0, 1.0, 0.0),
            expected_sum => 0.0,
            description => "All zero inputs                     "
        ),
        -- Test 5: Large values
        (
            inputs => (2.0, 2.0, 2.0, 2.0),
            weights => (2.0, 2.0, 2.0, 2.0, 0.0),
            expected_sum => 16.0,  -- 2*2 + 2*2 + 2*2 + 2*2 = 16
            description => "Large values                        "
        )
    );

begin
    -- DUT instantiation
    dut: entity work.neuron
        generic map(
            num_inputs => NUM_INPUTS_C,
            data_width => DATA_WIDTH_C,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            inputs_i => inputs_s,
            weights_i => weights_s,
            output_o => output_s,
            overflow_o => overflow_s
        );

    -- Convert signals for monitoring
    monitor_proc : process(inputs_s, weights_s, output_s)
        variable input_fixed  : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
        variable weight_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
        variable output_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
    begin
        for i in 0 to NUM_INPUTS_C - 1 loop
            input_fixed := to_sfixed(arg => inputs_s(i),
                                     left_index => input_fixed'high,
                                     right_index => input_fixed'low);
            inputs_real(i) <= to_real(input_fixed);
        end loop;

        for i in 0 to NUM_INPUTS_C loop
            weight_fixed := to_sfixed(arg => weights_s(i),
                                      left_index => weight_fixed'high,
                                      right_index => weight_fixed'low);
            weights_real(i) <= to_real(weight_fixed);
        end loop;

        output_fixed := to_sfixed(arg => output_s,
                                  left_index => output_fixed'high,
                                  right_index => output_fixed'low);
        output_real <= to_real(output_fixed);
    end process;

    -- Test process
    test_proc: process
        variable test_pass : boolean;
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
        variable expected_sigmoid : real;
        variable input_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
        variable weight_fixed : sfixed((DATA_WIDTH_C  + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
    begin
        report "========================================";
        report "Starting Neuron Test";
        report "Num Inputs: " & integer'image(NUM_INPUTS_C);
        report "Data Width: " & integer'image(DATA_WIDTH_C);
        if USE_SIGMOID_C then
            report "Activation: Sigmoid";
        else
            report "Activation: ReLU";
        end if;
        report "========================================";

        -- Run test cases
        for test_idx in TEST_CASES'range loop
            report "----------------------------------------";
            report "Test " & integer'image(test_idx) & ": " & TEST_CASES(test_idx).description;

            -- Set inputs
            for i in 0 to NUM_INPUTS_C - 1 loop
                input_fixed := to_sfixed(arg => TEST_CASES(test_idx).inputs(i),
                                        left_index => input_fixed'high,
                                        right_index => input_fixed'low);
                inputs_s(i) <= to_slv(input_fixed);
            end loop;

            -- Set weights (including bias)
            for i in 0 to NUM_INPUTS_C loop
                weight_fixed := to_sfixed(arg => TEST_CASES(test_idx).weights(i),
                                         left_index => weight_fixed'high,
                                         right_index => weight_fixed'low);
                weights_s(i) <= to_slv(weight_fixed);
            end loop;

            -- Wait for computation
            wait for 50 ns;

            -- Calculate expected output based on activation
            if USE_SIGMOID_C then
                -- Sigmoid activation
                expected_sigmoid := 1.0 / (1.0 + 2.718281828459 ** (-TEST_CASES(test_idx).expected_sum));
                test_pass := abs(output_real - expected_sigmoid) < 0.07;  -- 5% tolerance

                report "  Expected sum: " & real'image(TEST_CASES(test_idx).expected_sum);
                report "  Expected sigmoid: " & real'image(expected_sigmoid);
                report "  Actual output: " & real'image(output_real);
                report "  Error: " & real'image(abs(output_real - expected_sigmoid));
            else
                -- ReLU activation
                if TEST_CASES(test_idx).expected_sum < 0.0 then
                    test_pass := output_real < 0.1;  -- Should be ~0
                else
                    test_pass := abs(output_real - TEST_CASES(test_idx).expected_sum) < 0.1;
                end if;

                report "  Expected sum: " & real'image(TEST_CASES(test_idx).expected_sum);
                report "  Actual output: " & real'image(output_real);
            end if;

            if overflow_s = '1' then
                report "  WARNING: Overflow detected!";
            end if;

            if test_pass then
                report "  PASS";
                pass_count := pass_count + 1;
            else
                report "  FAIL";
                fail_count := fail_count + 1;
            end if;

            wait for 50 ns;
        end loop;

        -- Summary
        report "========================================";
        report "Test Summary:";
        report "  Passed: " & integer'image(pass_count) & "/" &
               integer'image(TEST_CASES'length);
        report "  Failed: " & integer'image(fail_count) & "/" &
               integer'image(TEST_CASES'length);
        report "========================================";

        if fail_count = 0 then
            report "ALL TESTS PASSED!" severity note;
        else
            report "SOME TESTS FAILED!" severity warning;
        end if;

        wait;
    end process test_proc;

end architecture testbench;

#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from config import SigmoidConfig, sigmoid

def get_sfixed_range(width, frac_width=None):
    if frac_width is None:
        high = (width + 1) // 2 - 1
        low = -(width // 2)
    else:
        high = width - frac_width - 1
        low = -frac_width
    return high, low

def generate_activation_func_tb(config : SigmoidConfig, output_file="testbench/activation_func_tb.vhd"):
    """Generate VHDL testbench for sigmoid activation function"""

    # Generate test vectors
    test_vectors = []
    x_min, x_max = config.input_range

    for i in range(config.num_test_inputs):
        x = x_min + (x_max - x_min) * i / (config.num_test_inputs - 1)
        expected = sigmoid(x)
        test_vectors.append((x, expected))

    in_high, in_low = get_sfixed_range(config.input_width, config.input_frac_width)
    out_high, out_low = get_sfixed_range(config.output_width)

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func_tb is
end entity activation_func_tb;

architecture testbench of activation_func_tb is
    -- Component declaration
    component activation_func is
        generic(
            input_width : integer := {config.input_width};
            input_frac_width : integer := {config.input_frac_width};
            output_width : integer := {config.output_width}
        );
        port(
            input_i : in std_logic_vector(input_width - 1 downto 0);
            output_o : out sfixed((output_width + 1) / 2 - 1 downto - (output_width / 2))
        );
    end component;

    -- Test signals
    signal input_s : std_logic_vector({config.input_width} - 1 downto 0);
    signal output_s : sfixed({out_high} downto {out_low});

    -- Helper signals
    signal input_sfixed : sfixed({in_high} downto {in_low});
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

    constant TOLERANCE : real := 0.01;  -- 1% tolerance for comparison

begin
    -- DUT instantiation (sigmoid architecture)
    dut: entity work.activation_func(sigmoid)
        generic map(
            input_width => {config.input_width},
            input_frac_width => {config.input_frac_width},
            output_width => {config.output_width}
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
        report "Input Width: " & integer'image({config.input_width});
        report "Output Width: " & integer'image({config.output_width});
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
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated testbench: {output_file}")
    return output_file

def generate_neuron_tb(config : SigmoidConfig, output_file="testbench/neuron_tb.vhd"):
    """Generate VHDL testbench for neuron"""

    # Calculate max step in LUT for tolerance
    # Max derivative of sigmoid is 0.25 at x=0
    # Step size in input is (max-min)/lut_size
    # Max step in output is approx 0.25 * input_step
    input_step = (config.input_range[1] - config.input_range[0]) / config.lut_size
    max_lut_step = 0.25 * input_step
    # Add some safety margin
    tolerance = max_lut_step * 1.5

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.math_real.all;

use work.types.all;

entity neuron_tb is
end entity neuron_tb;

architecture testbench of neuron_tb is
    constant NUM_INPUTS_C : integer := 4;
    constant DATA_WIDTH_C : integer := {config.input_width};
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := {tolerance:.6f};

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
        report "Tolerance: " & real'image(TOLERANCE_C);
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
                test_pass := abs(output_real - expected_sigmoid) < TOLERANCE_C;

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
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated neuron testbench: {output_file}")
    return output_file

def generate_layer_tb(config : SigmoidConfig, output_file="testbench/layer_tb.vhd"):
    """Generate VHDL testbench for layer"""

    # Calculate tolerance
    input_step = (config.input_range[1] - config.input_range[0]) / config.lut_size
    max_lut_step = 0.25 * input_step
    tolerance = max_lut_step * 2.0 # Slightly looser for layer due to accumulation

    # Python simulation for expected values
    def sigmoid_activation(x):
        return 1.0 / (1.0 + math.exp(-x))

    # Test 1: Single Layer (4 inputs -> 2 outputs)
    inputs_1 = [0.5, 1.0, -0.5, 0.25]
    weights_1 = [
        [1.0, 0.5, -0.5, 0.75, 0.5],   # Neuron 0 (last is bias)
        [0.5, 1.0, 0.25, -0.5, -0.25]  # Neuron 1
    ]
    
    expected_1 = []
    for i in range(2):
        sum_val = weights_1[i][4] # Bias
        for j in range(4):
            sum_val += inputs_1[j] * weights_1[i][j]
        expected_1.append(sigmoid_activation(sum_val))

    # Test 2: Two Layers (4 -> 2 -> 1)
    inputs_2 = [1.0, 0.5, -1.0, 0.75]
    weights_2_L1 = [
        [0.8, 0.3, -0.4, 0.6, 0.2],    # Neuron 0
        [0.4, 0.9, 0.1, -0.3, -0.1]    # Neuron 1
    ]
    weights_2_L2 = [
        [1.2, 0.7, 0.3]                # Neuron 0 (2 inputs + bias)
    ]

    # Layer 1
    L1_out = []
    for i in range(2):
        sum_val = weights_2_L1[i][4]
        for j in range(4):
            sum_val += inputs_2[j] * weights_2_L1[i][j]
        L1_out.append(sigmoid_activation(sum_val))

    # Layer 2
    L2_out = []
    for i in range(1):
        sum_val = weights_2_L2[i][2]
        for j in range(2):
            sum_val += L1_out[j] * weights_2_L2[i][j]
        L2_out.append(sigmoid_activation(sum_val))


    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity layer_tb is
end entity layer_tb;

architecture testbench of layer_tb is
    -- Test 1 configuration: Single layer (4 inputs -> 2 outputs)
    constant NUM_INPUTS_1 : integer := 4;
    constant NUM_OUTPUTS_1 : integer := 2;
    constant DATA_WIDTH_C : integer := {config.input_width};
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := {tolerance:.6f};

    -- Test 2 configuration: Two layers (4 -> 2 -> 1)
    constant NUM_INPUTS_2 : integer := 4;
    constant NUM_OUTPUTS_2 : integer := 2;
    constant NUM_OUTPUTS_3 : integer := 1;

    -- Clock period
    constant CLK_PERIOD : time := 10 ns;

    -- Clock signal
    signal clk : std_logic := '0';
    signal stop_clock : boolean := false;

    -- Test 1 signals
    signal inputs_1 : std_logic_bus_array(NUM_INPUTS_1 - 1 downto 0)(DATA_WIDTH_C - 1 downto 0);
    signal weights_1 : weights_matrix(0 to NUM_OUTPUTS_1 - 1)(0 to NUM_INPUTS_1)(DATA_WIDTH_C - 1 downto 0);
    signal output_1 : std_logic_bus_array(0 to NUM_OUTPUTS_1 - 1)(DATA_WIDTH_C - 1 downto 0);

    -- Test 2 signals
    signal inputs_2_layer1 : std_logic_bus_array(NUM_INPUTS_2 - 1 downto 0)(DATA_WIDTH_C - 1 downto 0);
    signal weights_2_layer1 : weights_matrix(0 to NUM_OUTPUTS_2 - 1)(0 to NUM_INPUTS_2)(DATA_WIDTH_C - 1 downto 0);
    signal output_2_layer1 : std_logic_bus_array(0 to NUM_OUTPUTS_2 - 1)(DATA_WIDTH_C - 1 downto 0);

    signal weights_2_layer2 : weights_matrix(0 to NUM_OUTPUTS_3 - 1)(0 to NUM_OUTPUTS_2)(DATA_WIDTH_C - 1 downto 0);
    signal output_2_layer2 : std_logic_bus_array(0 to NUM_OUTPUTS_3 - 1)(DATA_WIDTH_C - 1 downto 0);

    -- Helper signals for monitoring (convert to real for display)
    type real_array is array (integer range <>) of real;
    type real_array_2d is array (integer range<>, integer range<>) of real;
    signal inputs_1_real : real_array(NUM_INPUTS_1 - 1 downto 0);
    signal output_1_real : real_array(0 to NUM_OUTPUTS_1 - 1);
    signal inputs_2_real : real_array(NUM_INPUTS_2 - 1 downto 0);
    signal output_2_layer1_real : real_array(0 to NUM_OUTPUTS_2 - 1);
    signal output_2_layer2_real : real_array(0 to NUM_OUTPUTS_3 - 1);

begin
    -- ========================================================================
    -- Clock generation
    -- ========================================================================
    clk_proc: process
    begin
        while not stop_clock loop
            clk <= '0';
            wait for CLK_PERIOD / 2;
            clk <= '1';
            wait for CLK_PERIOD / 2;
        end loop;
        wait;
    end process clk_proc;

    -- ========================================================================
    -- Test 1: Single Layer (4 inputs -> 2 outputs)
    -- ========================================================================
    layer_1_inst: entity work.layer
        generic map(
            num_inputs => NUM_INPUTS_1,
            num_outputs => NUM_OUTPUTS_1,
            data_width => DATA_WIDTH_C,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            inputs_i => inputs_1,
            weights_matrix_i => weights_1,
            output_o => output_1
        );

    -- ========================================================================
    -- Test 2: Two Layers (4 -> 2 -> 1)
    -- ========================================================================
    layer_2_1_inst: entity work.layer
        generic map(
            num_inputs => NUM_INPUTS_2,
            num_outputs => NUM_OUTPUTS_2,
            data_width => DATA_WIDTH_C,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            inputs_i => inputs_2_layer1,
            weights_matrix_i => weights_2_layer1,
            output_o => output_2_layer1
        );

    layer_2_2_inst: entity work.layer
        generic map(
            num_inputs => NUM_OUTPUTS_2,
            num_outputs => NUM_OUTPUTS_3,
            data_width => DATA_WIDTH_C,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            inputs_i => output_2_layer1,
            weights_matrix_i => weights_2_layer2,
            output_o => output_2_layer2
        );

    -- ========================================================================
    -- Monitor process: Convert signals to real for display
    -- ========================================================================
    monitor_proc: process(inputs_1, output_1, inputs_2_layer1, output_2_layer1, output_2_layer2)
        variable temp_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
    begin
        -- Test 1 monitoring
        for i in 0 to NUM_INPUTS_1 - 1 loop
            temp_fixed := to_sfixed(inputs_1(i), temp_fixed'high, temp_fixed'low);
            inputs_1_real(i) <= to_real(temp_fixed);
        end loop;

        for i in 0 to NUM_OUTPUTS_1 - 1 loop
            temp_fixed := to_sfixed(output_1(i), temp_fixed'high, temp_fixed'low);
            output_1_real(i) <= to_real(temp_fixed);
        end loop;

        -- Test 2 monitoring
        for i in 0 to NUM_INPUTS_2 - 1 loop
            temp_fixed := to_sfixed(inputs_2_layer1(i), temp_fixed'high, temp_fixed'low);
            inputs_2_real(i) <= to_real(temp_fixed);
        end loop;

        for i in 0 to NUM_OUTPUTS_2 - 1 loop
            temp_fixed := to_sfixed(output_2_layer1(i), temp_fixed'high, temp_fixed'low);
            output_2_layer1_real(i) <= to_real(temp_fixed);
        end loop;

        for i in 0 to NUM_OUTPUTS_3 - 1 loop
            temp_fixed := to_sfixed(output_2_layer2(i), temp_fixed'high, temp_fixed'low);
            output_2_layer2_real(i) <= to_real(temp_fixed);
        end loop;
    end process monitor_proc;

    -- ========================================================================
    -- Test stimulus process
    -- ========================================================================
    test_proc: process
        variable input_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
        variable weight_fixed : sfixed((DATA_WIDTH_C + 1) / 2 - 1 downto -(DATA_WIDTH_C / 2));
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
        variable total_tests : integer := 0;
    begin
        report "========================================";
        report "Starting Layer Testbench";
        report "Data Width: " & integer'image(DATA_WIDTH_C);
        report "Activation: Sigmoid";
        report "Tolerance: " & real'image(TOLERANCE_C);
        report "========================================";

        -- ====================================================================
        -- TEST 1: Single Layer (4 inputs -> 2 outputs)
        -- ====================================================================
        report "========================================";
        report "TEST 1: Single Layer (4 inputs -> 2 outputs)";
        report "========================================";

        -- Initialize inputs for Test 1
        -- Inputs: {inputs_1}
        input_fixed := to_sfixed({inputs_1[0]}, input_fixed'high, input_fixed'low);
        inputs_1(0) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_1[1]}, input_fixed'high, input_fixed'low);
        inputs_1(1) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_1[2]}, input_fixed'high, input_fixed'low);
        inputs_1(2) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_1[3]}, input_fixed'high, input_fixed'low);
        inputs_1(3) <= to_slv(input_fixed);

        -- Set weights for Test 1
        -- Neuron 0: weights={weights_1[0][:-1]}, bias={weights_1[0][-1]}
        -- Neuron 1: weights={weights_1[1][:-1]}, bias={weights_1[1][-1]}

        -- Neuron 0 weights
        weight_fixed := to_sfixed({weights_1[0][0]}, weight_fixed'high, weight_fixed'low);
        weights_1(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[0][1]}, weight_fixed'high, weight_fixed'low);
        weights_1(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[0][2]}, weight_fixed'high, weight_fixed'low);
        weights_1(0)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[0][3]}, weight_fixed'high, weight_fixed'low);
        weights_1(0)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[0][4]}, weight_fixed'high, weight_fixed'low);  -- bias
        weights_1(0)(4) <= to_slv(weight_fixed);

        -- Neuron 1 weights
        weight_fixed := to_sfixed({weights_1[1][0]}, weight_fixed'high, weight_fixed'low);
        weights_1(1)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[1][1]}, weight_fixed'high, weight_fixed'low);
        weights_1(1)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[1][2]}, weight_fixed'high, weight_fixed'low);
        weights_1(1)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[1][3]}, weight_fixed'high, weight_fixed'low);
        weights_1(1)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_1[1][4]}, weight_fixed'high, weight_fixed'low);  -- bias
        weights_1(1)(4) <= to_slv(weight_fixed);

        report "Inputs set: {inputs_1}";
        report "Weights configured for 2 neurons";
        report "Waiting 2 clock cycles for output...";

        -- Wait 2 clock cycles for pipeline
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 4;  -- Wait a bit into the cycle for signals to settle

        report "Test 1 Output:";
        report "  Output[0] = " & real'image(output_1_real(0)) & " (Expected: {expected_1[0]:.6f})";
        report "  Output[1] = " & real'image(output_1_real(1)) & " (Expected: {expected_1[1]:.6f})";
        
        total_tests := total_tests + 2;
        
        if abs(output_1_real(0) - {expected_1[0]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1;
            report "  Output[0] PASS";
        else
            fail_count := fail_count + 1;
            report "  Output[0] FAIL";
        end if;

        if abs(output_1_real(1) - {expected_1[1]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1;
            report "  Output[1] PASS";
        else
            fail_count := fail_count + 1;
            report "  Output[1] FAIL";
        end if;

        wait for CLK_PERIOD * 2;

        -- ====================================================================
        -- TEST 2: Two Layers (4 -> 2 -> 1)
        -- ====================================================================
        report "========================================";
        report "TEST 2: Two Layers (4 -> 2 -> 1)";
        report "========================================";

        -- Initialize inputs for Test 2
        -- Inputs: {inputs_2}
        input_fixed := to_sfixed({inputs_2[0]}, input_fixed'high, input_fixed'low);
        inputs_2_layer1(0) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_2[1]}, input_fixed'high, input_fixed'low);
        inputs_2_layer1(1) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_2[2]}, input_fixed'high, input_fixed'low);
        inputs_2_layer1(2) <= to_slv(input_fixed);

        input_fixed := to_sfixed({inputs_2[3]}, input_fixed'high, input_fixed'low);
        inputs_2_layer1(3) <= to_slv(input_fixed);

        -- Set weights for Layer 1 (4 inputs -> 2 outputs)
        -- Neuron 0: weights={weights_2_L1[0][:-1]}, bias={weights_2_L1[0][-1]}
        -- Neuron 1: weights={weights_2_L1[1][:-1]}, bias={weights_2_L1[1][-1]}

        -- Layer 1, Neuron 0 weights
        weight_fixed := to_sfixed({weights_2_L1[0][0]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[0][1]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[0][2]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[0][3]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[0][4]}, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer1(0)(4) <= to_slv(weight_fixed);

        -- Layer 1, Neuron 1 weights
        weight_fixed := to_sfixed({weights_2_L1[1][0]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[1][1]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[1][2]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[1][3]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L1[1][4]}, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer1(1)(4) <= to_slv(weight_fixed);

        -- Set weights for Layer 2 (2 inputs -> 1 output)
        -- Neuron 0: weights={weights_2_L2[0][:-1]}, bias={weights_2_L2[0][-1]}

        weight_fixed := to_sfixed({weights_2_L2[0][0]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer2(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L2[0][1]}, weight_fixed'high, weight_fixed'low);
        weights_2_layer2(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed({weights_2_L2[0][2]}, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer2(0)(2) <= to_slv(weight_fixed);

        report "Inputs set: {inputs_2}";
        report "Weights configured for Layer 1 (4->2) and Layer 2 (2->1)";
        report "Waiting 4 clock cycles for final output...";

        -- Wait 4 clock cycles for both layers
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        report "After 2 cycles - Layer 1 output:";
        report "  Layer1 Output[0] = " & real'image(output_2_layer1_real(0)) & " (Expected: {L1_out[0]:.6f})";
        report "  Layer1 Output[1] = " & real'image(output_2_layer1_real(1)) & " (Expected: {L1_out[1]:.6f})";

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 4;  -- Wait a bit into the cycle for signals to settle

        report "After 4 cycles - Final output:";
        report "  Layer2 Output[0] = " & real'image(output_2_layer2_real(0)) & " (Expected: {L2_out[0]:.6f})";
        
        total_tests := total_tests + 1;
        if abs(output_2_layer2_real(0) - {L2_out[0]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1;
            report "  Final Output PASS";
        else
            fail_count := fail_count + 1;
            report "  Final Output FAIL";
        end if;

        wait for CLK_PERIOD * 2;

        -- ====================================================================
        -- Test Complete
        -- ====================================================================
        report "========================================";
        report "Test Summary:";
        report "  Passed: " & integer'image(pass_count) & "/" & integer'image(total_tests);
        report "  Failed: " & integer'image(fail_count) & "/" & integer'image(total_tests);
        report "========================================";

        if fail_count = 0 then
            report "ALL TESTS PASSED!" severity note;
        else
            report "SOME TESTS FAILED!" severity warning;
        end if;

        stop_clock <= true;
        wait;
    end process test_proc;

end architecture testbench;
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated layer testbench: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Generate testbenches for VHDL components'
    )
    parser.add_argument('--lut-size', type=int, default=256,
                        help='Number of LUT entries (default: 256)')
    parser.add_argument('--input-width', type=int, default=32,
                        help='Input width in bits (default: 32)')
    parser.add_argument('--input-frac-width', type=int, default=16,
                        help='Input fractional width in bits (default: 16)')
    parser.add_argument('--output-width', type=int, default=32,
                        help='Output width in bits (default: 32)')
    parser.add_argument('--num-tests', type=int, default=16,
                        help='Number of test vectors (default: 16)')
    parser.add_argument('--input-min', type=float, default=-8.0,
                        help='Minimum input value (default: -8.0)')
    parser.add_argument('--input-max', type=float, default=8.0,
                        help='Maximum input value (default: 8.0)')

    args = parser.parse_args()

    # Create configuration
    config = SigmoidConfig(
        lut_size=args.lut_size,
        input_width=args.input_width,
        input_frac_width=args.input_frac_width,
        output_width=args.output_width,
        num_test_inputs=args.num_tests,
        input_range=(args.input_min, args.input_max)
    )

    print("=" * 60)
    print("Testbench Generation")
    print("=" * 60)
    
    # Generate activation function testbench
    print("\n[1/3] Generating activation function testbench...")
    generate_activation_func_tb(config)
    
    # Generate neuron testbench
    print("\n[2/3] Generating neuron testbench...")
    generate_neuron_tb(config)
    
    # Generate layer testbench
    print("\n[3/3] Generating layer testbench...")
    generate_layer_tb(config)

    print("\n" + "=" * 60)
    print("Testbench Generation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

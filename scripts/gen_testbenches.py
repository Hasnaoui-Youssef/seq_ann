#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from config import SigmoidConfig, sigmoid

def get_sfixed_range(config):
    """Get sfixed range using INT_BITS and FRAC_BITS from config"""
    return config.int_bits - 1, -config.frac_bits

def generate_activation_func_tb(config : SigmoidConfig, output_file="testbench/activation_func_tb.vhd"):
    """Generate VHDL testbench for sigmoid activation function"""

    # Generate test vectors
    test_vectors = []
    x_min, x_max = config.input_range

    for i in range(config.num_test_inputs):
        x = x_min + (x_max - x_min) * i / (config.num_test_inputs - 1)
        expected = sigmoid(x)
        test_vectors.append((x, expected))

    sfixed_high, sfixed_low = get_sfixed_range(config)

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
            input_width : integer := {config.data_width};
            input_frac_width : integer := {config.frac_bits};
            output_width : integer := {config.data_width}
        );
        port(
            input_i : in std_logic_vector(input_width - 1 downto 0);
            output_o : out sfixed(INT_BITS - 1 downto -FRAC_BITS)
        );
    end component;

    -- Test signals
    signal input_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal output_s : sfixed(INT_BITS - 1 downto -FRAC_BITS);

    -- Helper signals
    signal input_sfixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
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
            input_width => DATA_WIDTH,
            input_frac_width => FRAC_BITS,
            output_width => DATA_WIDTH
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
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated testbench: {output_file}")
    return output_file

def generate_neuron_tb(config : SigmoidConfig, output_file="testbench/neuron_tb.vhd"):
    """Generate VHDL testbench for neuron with backpropagation interface"""

    # Calculate max step in LUT for tolerance
    input_step = (config.input_range[1] - config.input_range[0]) / config.lut_size
    max_lut_step = 0.25 * input_step
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
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := {tolerance:.6f};
    constant CLK_PERIOD : time := 10 ns;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';
    
    -- Forward pass signals
    signal fwd_en : std_logic := '0';
    signal inputs_s : std_logic_bus_array(0 to NUM_INPUTS_C - 1)(DATA_WIDTH - 1 downto 0);
    signal weights_s : std_logic_bus_array(0 to NUM_INPUTS_C - 1)(DATA_WIDTH - 1 downto 0);
    signal bias_s : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal output_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    
    -- Backward pass signals (not used in forward-only test)
    signal bwd_en : std_logic := '0';
    signal error_s : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal grad_weights_s : std_logic_bus_array(0 to NUM_INPUTS_C - 1)(DATA_WIDTH - 1 downto 0);
    signal grad_bias_s : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal grad_inputs_s : std_logic_bus_array(0 to NUM_INPUTS_C - 1)(DATA_WIDTH - 1 downto 0);

    -- Test vectors
    type real_array is array (integer range <>) of real;
    type test_case is record
        inputs : real_array(0 to NUM_INPUTS_C - 1);
        weights : real_array(0 to NUM_INPUTS_C);  -- Last is bias
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
    -- Clock generation
    process
    begin
        while true loop
            clk <= '0'; wait for CLK_PERIOD / 2;
            clk <= '1'; wait for CLK_PERIOD / 2;
        end loop;
    end process;

    -- DUT instantiation
    dut: entity work.neuron
        generic map(
            num_inputs => NUM_INPUTS_C,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            rst => rst,
            fwd_en => fwd_en,
            inputs_i => inputs_s,
            weights_i => weights_s,
            bias_i => bias_s,
            output_o => output_s,
            bwd_en => bwd_en,
            error_i => error_s,
            grad_weights_o => grad_weights_s,
            grad_bias_o => grad_bias_s,
            grad_inputs_o => grad_inputs_s
        );

    -- Test process
    test_proc: process
        variable test_pass : boolean;
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
        variable expected_sigmoid : real;
        variable output_real : real;
        variable input_fixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
        variable weight_fixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
        variable output_fixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
    begin
        report "========================================";
        report "Starting Neuron Test";
        report "Num Inputs: " & integer'image(NUM_INPUTS_C);
        report "Data Width: " & integer'image(DATA_WIDTH);
        if USE_SIGMOID_C then
            report "Activation: Sigmoid";
        else
            report "Activation: ReLU";
        end if;
        report "Tolerance: " & real'image(TOLERANCE_C);
        report "========================================";
        
        -- Reset
        rst <= '1';
        fwd_en <= '0';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD * 2;

        -- Run test cases
        for test_idx in TEST_CASES'range loop
            report "----------------------------------------";
            report "Test " & integer'image(test_idx) & ": " & TEST_CASES(test_idx).description;
            
            -- Set weights (directly, no weight loading interface)
            for i in 0 to NUM_INPUTS_C - 1 loop
                weight_fixed := to_sfixed(arg => TEST_CASES(test_idx).weights(i),
                                         left_index => weight_fixed'high,
                                         right_index => weight_fixed'low);
                weights_s(i) <= to_slv(weight_fixed);
            end loop;
            -- Set bias
            weight_fixed := to_sfixed(arg => TEST_CASES(test_idx).weights(NUM_INPUTS_C),
                                     left_index => weight_fixed'high,
                                     right_index => weight_fixed'low);
            bias_s <= to_slv(weight_fixed);
            
            -- Set inputs
            for i in 0 to NUM_INPUTS_C - 1 loop
                input_fixed := to_sfixed(arg => TEST_CASES(test_idx).inputs(i),
                                        left_index => input_fixed'high,
                                        right_index => input_fixed'low);
                inputs_s(i) <= to_slv(input_fixed);
            end loop;

            -- Enable forward pass
            fwd_en <= '1';
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);  -- Wait for pipeline
            wait for CLK_PERIOD / 2; -- Sample in middle of cycle

            -- Capture output
            output_fixed := to_sfixed(output_s, output_fixed);
            output_real := to_real(output_fixed);

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

            if test_pass then
                report "  PASS";
                pass_count := pass_count + 1;
            else
                report "  FAIL";
                fail_count := fail_count + 1;
            end if;

            fwd_en <= '0';
            wait for CLK_PERIOD;
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

    # Test Configuration: Single Layer (4 inputs -> 2 outputs)
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

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity layer_tb is
end entity layer_tb;

architecture testbench of layer_tb is
    -- Configuration
    constant NUM_INPUTS : integer := 4;
    constant NUM_OUTPUTS : integer := 2;
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := {tolerance:.6f};
    constant CLK_PERIOD : time := 10 ns;

    -- Signals
    signal clk : std_logic := '0';
    signal inputs_s : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
    signal output_s : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0);
    
    -- Weight loading
    signal load_enable : std_logic := '0';
    signal neuron_select : integer := 0;
    signal weight_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_index : integer := 0;

begin
    -- Clock generation
    process
    begin
        while true loop
            clk <= '0'; wait for CLK_PERIOD / 2;
            clk <= '1'; wait for CLK_PERIOD / 2;
        end loop;
    end process;

    -- DUT Instantiation
    dut: entity work.layer
        generic map(
            num_inputs => NUM_INPUTS,
            layer_size => NUM_OUTPUTS,
            use_sigmoid => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            inputs_i => inputs_s,
            load_enable => load_enable,
            neuron_select => neuron_select,
            weight_data => weight_data,
            weight_index => weight_index,
            output_o => output_s
        );

    -- Test Process
    test_proc: process
        variable input_fixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
        variable weight_fixed : sfixed(INT_BITS - 1 downto -FRAC_BITS);
        variable output_real : real;
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
    begin
        report "========================================";
        report "Starting Layer Testbench";
        report "========================================";
        
        -- Init
        load_enable <= '0';
        wait for CLK_PERIOD * 2;

        -- 1. Load Weights
        report "Loading weights...";
        
        -- Neuron 0
        neuron_select <= 0;
        -- Weights: {weights_1[0]}
        -- w0
        weight_fixed := to_sfixed({weights_1[0][0]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 0; load_enable <= '1'; wait until rising_edge(clk);
        -- w1
        weight_fixed := to_sfixed({weights_1[0][1]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 1; load_enable <= '1'; wait until rising_edge(clk);
        -- w2
        weight_fixed := to_sfixed({weights_1[0][2]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 2; load_enable <= '1'; wait until rising_edge(clk);
        -- w3
        weight_fixed := to_sfixed({weights_1[0][3]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 3; load_enable <= '1'; wait until rising_edge(clk);
        -- bias
        weight_fixed := to_sfixed({weights_1[0][4]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 4; load_enable <= '1'; wait until rising_edge(clk);

        -- Neuron 1
        neuron_select <= 1;
        -- Weights: {weights_1[1]}
        -- w0
        weight_fixed := to_sfixed({weights_1[1][0]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 0; load_enable <= '1'; wait until rising_edge(clk);
        -- w1
        weight_fixed := to_sfixed({weights_1[1][1]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 1; load_enable <= '1'; wait until rising_edge(clk);
        -- w2
        weight_fixed := to_sfixed({weights_1[1][2]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 2; load_enable <= '1'; wait until rising_edge(clk);
        -- w3
        weight_fixed := to_sfixed({weights_1[1][3]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 3; load_enable <= '1'; wait until rising_edge(clk);
        -- bias
        weight_fixed := to_sfixed({weights_1[1][4]}, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 4; load_enable <= '1'; wait until rising_edge(clk);

        load_enable <= '0';
        wait until rising_edge(clk);
        report "Weights loaded.";

        -- 2. Set Inputs
        -- Inputs: {inputs_1}
        input_fixed := to_sfixed({inputs_1[0]}, input_fixed'high, input_fixed'low);
        inputs_s(0) <= to_slv(input_fixed);
        input_fixed := to_sfixed({inputs_1[1]}, input_fixed'high, input_fixed'low);
        inputs_s(1) <= to_slv(input_fixed);
        input_fixed := to_sfixed({inputs_1[2]}, input_fixed'high, input_fixed'low);
        inputs_s(2) <= to_slv(input_fixed);
        input_fixed := to_sfixed({inputs_1[3]}, input_fixed'high, input_fixed'low);
        inputs_s(3) <= to_slv(input_fixed);

        -- 3. Wait for computation
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 2;

        -- 4. Check Outputs
        -- Output 0
        output_real := to_real(to_sfixed(output_s(0), input_fixed));
        report "Output[0]: " & real'image(output_real) & " Expected: {expected_1[0]:.6f}";
        if abs(output_real - {expected_1[0]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1; report "PASS";
        else
            fail_count := fail_count + 1; report "FAIL";
        end if;

        -- Output 1
        output_real := to_real(to_sfixed(output_s(1), input_fixed));
        report "Output[1]: " & real'image(output_real) & " Expected: {expected_1[1]:.6f}";
        if abs(output_real - {expected_1[1]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1; report "PASS";
        else
            fail_count := fail_count + 1; report "FAIL";
        end if;

        -- Summary
        report "========================================";
        report "Passed: " & integer'image(pass_count) & "/2";
        report "Failed: " & integer'image(fail_count) & "/2";
        
        if fail_count = 0 then
            report "ALL TESTS PASSED!" severity note;
        else
            report "SOME TESTS FAILED!" severity warning;
        end if;

        wait;
    end process;

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
    parser.add_argument('--data-width', type=int, default=32,
                        help='Data width in bits (default: 32)')
    parser.add_argument('--frac-bits', type=int, default=16,
                        help='Fractional bits (default: 16)')
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
        data_width=args.data_width,
        frac_bits=args.frac_bits,
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

#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import sys

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from config import auto_load_config, SigmoidConfig as YamlSigmoidConfig


def sigmoid(x: float) -> float:
    """Compute sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))

def get_sfixed_range(config):
    """Get sfixed range using INT_BITS and FRAC_BITS from config"""
    return config.int_bits - 1, -config.frac_bits

def generate_activation_func_tb(sigmoid_config: YamlSigmoidConfig, int_bits: int, frac_bits: int, 
                                num_test_inputs: int, output_file: str = "testbench/activation_func_tb.vhd"):
    """Generate VHDL testbench for sigmoid activation function"""

    # Generate test vectors
    test_vectors = []
    x_min, x_max = sigmoid_config.input_range

    for i in range(num_test_inputs):
        x = x_min + (x_max - x_min) * i / (num_test_inputs - 1)
        expected = sigmoid(x)
        test_vectors.append((x, expected))

    data_width = int_bits + frac_bits

    vhdl_code = f"""library ieee;
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
    signal output_s : sfixed_bus;

    -- Helper signals
    signal input_sfixed : sfixed_bus;
    signal input_real : real;
    signal output_real : real;

    -- Test configuration
    constant NUM_TESTS : integer := {num_test_inputs};
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

    vhdl_code += """    );

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
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated testbench: {output_file}")
    return output_file

def generate_neuron_tb(sigmoid_config: YamlSigmoidConfig, int_bits: int, frac_bits: int,
                       output_file: str = "testbench/neuron_tb.vhd"):
    """Generate VHDL testbench for neuron with backpropagation interface"""

    # Calculate max step in LUT for tolerance
    input_step = (sigmoid_config.input_range[1] - sigmoid_config.input_range[0]) / sigmoid_config.lut_size
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
        variable input_fixed : sfixed_bus;
        variable weight_fixed : sfixed_bus;
        variable output_fixed : sfixed_bus;
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

def generate_layer_tb(sigmoid_config: YamlSigmoidConfig, int_bits: int, frac_bits: int,
                      output_file: str = "testbench/layer_tb.vhd"):
    """Generate VHDL testbench for layer"""

    # Calculate tolerance
    input_step = (sigmoid_config.input_range[1] - sigmoid_config.input_range[0]) / sigmoid_config.lut_size
    max_lut_step = 0.25 * input_step
    tolerance = max_lut_step * 2.0 # Slightly looser for layer due to accumulation

    # Python simulation for expected values
    def sigmoid_activation(x):
        return 1.0 / (1.0 + math.exp(-x))

    # Test Configuration: Single Layer (4 inputs -> 2 outputs)
    num_inputs = 4
    num_outputs = 2
    inputs_1 = [0.5, 1.0, -0.5, 0.25]
    # Weights in flat array format: [n0_w0, n0_w1, n0_w2, n0_w3, n0_bias, n1_w0, n1_w1, n1_w2, n1_w3, n1_bias]
    weights_flat = [1.0, 0.5, -0.5, 0.75, 0.5, 0.5, 1.0, 0.25, -0.5, -0.25]

    expected_1 = []
    for i in range(num_outputs):
        base = i * (num_inputs + 1)
        sum_val = weights_flat[base + num_inputs]  # Bias
        for j in range(num_inputs):
            sum_val += inputs_1[j] * weights_flat[base + j]
        expected_1.append(sigmoid_activation(sum_val))

    # Generate weight initialization code
    weight_init_code = ""
    for i, w in enumerate(weights_flat):
        weight_init_code += f"        weights_s({i}) <= to_slv(to_sfixed({w}, INT_BITS - 1, -FRAC_BITS));\n"

    # Generate input initialization code
    input_init_code = ""
    for i, inp in enumerate(inputs_1):
        input_init_code += f"        inputs_s({i}) <= to_slv(to_sfixed({inp}, INT_BITS - 1, -FRAC_BITS));\n"

    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.pkg_layer.all;

entity layer_tb is
end entity layer_tb;

architecture testbench of layer_tb is
    -- Configuration
    constant NUM_INPUTS : integer := {num_inputs};
    constant NUM_OUTPUTS : integer := {num_outputs};
    constant NUM_WEIGHTS : integer := (NUM_INPUTS + 1) * NUM_OUTPUTS;
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := {tolerance:.6f};
    constant CLK_PERIOD : time := 10 ns;

    -- Signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    -- Forward interface
    signal fwd_en : std_logic := '1';
    signal bwd_en : std_logic := '0';
    signal fwd_ctrl_in : layer_control_t := (valid => '0', last => '0');
    signal fwd_ctrl_out : layer_control_t;
    signal inputs_s : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));
    signal outputs_s : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0);

    -- Backward interface (unused)
    signal bwd_ctrl_in : layer_control_t := (valid => '0', last => '0');
    signal bwd_ctrl_out : layer_control_t;
    signal bwd_error_in : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));
    signal bwd_error_out : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);

    -- Weights
    signal weights_s : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));
    signal grads_s : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

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
            NUM_INPUTS => NUM_INPUTS,
            LAYER_SIZE => NUM_OUTPUTS,
            USE_SIGMOID => USE_SIGMOID_C
        )
        port map(
            clk => clk,
            rst => rst,
            fwd_en => fwd_en,
            bwd_en => bwd_en,
            fwd_ctrl_in => fwd_ctrl_in,
            fwd_data_in => inputs_s,
            fwd_ctrl_out => fwd_ctrl_out,
            fwd_data_out => outputs_s,
            bwd_ctrl_in => bwd_ctrl_in,
            bwd_error_in => bwd_error_in,
            bwd_ctrl_out => bwd_ctrl_out,
            bwd_error_out => bwd_error_out,
            weights_in => weights_s,
            grads_out => grads_s
        );

    -- Test Process
    test_proc: process
        variable output_real : real;
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
    begin
        report "========================================";
        report "Starting Layer Testbench";
        report "========================================";

        -- Reset
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD;

        -- Initialize weights
        report "Setting weights...";
{weight_init_code}
        -- Set inputs
        report "Setting inputs...";
{input_init_code}
        -- Trigger forward pass
        fwd_ctrl_in.valid <= '1';
        fwd_ctrl_in.last <= '1';
        wait until rising_edge(clk);
        fwd_ctrl_in.valid <= '0';
        fwd_ctrl_in.last <= '0';

        -- Wait for computation (pipeline delay)
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 2;

        -- Check Outputs
        -- Output 0
        output_real := to_real(to_sfixed(outputs_s(0), INT_BITS - 1, -FRAC_BITS));
        report "Output[0]: " & real'image(output_real) & " Expected: {expected_1[0]:.6f}";
        if abs(output_real - {expected_1[0]:.6f}) < TOLERANCE_C then
            pass_count := pass_count + 1; report "PASS";
        else
            fail_count := fail_count + 1; report "FAIL";
        end if;

        -- Output 1
        output_real := to_real(to_sfixed(outputs_s(1), INT_BITS - 1, -FRAC_BITS));
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
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (searches for *.nn_conf.yaml if not specified)')
    parser.add_argument('--root', type=str, default='.',
                        help='Project root directory (default: current directory)')
    parser.add_argument('--num-tests', type=int, default=16,
                        help='Number of test vectors (default: 16)')

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    # Load configuration
    try:
        config = auto_load_config(root_dir, args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    print("=" * 60)
    print("Testbench Generation")
    print("=" * 60)

    tb_dir = root_dir / "testbench"

    # Generate activation function testbench
    print("\n[1/3] Generating activation function testbench...")
    generate_activation_func_tb(
        config.sigmoid, config.int_bits, config.frac_bits,
        args.num_tests, str(tb_dir / "activation_func_tb.vhd")
    )

    # Generate neuron testbench
    print("\n[2/3] Generating neuron testbench...")
    generate_neuron_tb(
        config.sigmoid, config.int_bits, config.frac_bits,
        str(tb_dir / "neuron_tb.vhd")
    )

    # Generate layer testbench
    print("\n[3/3] Generating layer testbench...")
    generate_layer_tb(
        config.sigmoid, config.int_bits, config.frac_bits,
        str(tb_dir / "layer_tb.vhd")
    )

    print("\n" + "=" * 60)
    print("Testbench Generation Complete!")
    print("=" * 60)
    return 0

    print("\n" + "=" * 60)
    print("Testbench Generation Complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

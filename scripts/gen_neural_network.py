#!/usr/bin/env python3

"""
Neural Network Testbench Generator

This script generates VHDL testbenches for neural networks using Keras/TensorFlow.
It trains or loads a Keras model, extracts weights and biases, and generates
a complete VHDL testbench that:
1. Loads weights into the hardware neural network
2. Runs test vectors through both software and hardware models
3. Compares and validates results
"""

import argparse
import numpy as np
from pathlib import Path
from config import SigmoidConfig

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not available. Please install: pip install tensorflow")


def create_simple_model(layer_sizes, input_dim):
    """
    Create a simple feedforward neural network with sigmoid activation.
    
    Args:
        layer_sizes: List of integers specifying neurons in each hidden/output layer
        input_dim: Number of input features
    
    Returns:
        Keras Sequential model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required")
    
    model = keras.Sequential()
    
    # First hidden layer
    model.add(keras.layers.Dense(layer_sizes[0], activation='sigmoid', input_dim=input_dim))
    
    # Additional layers
    for size in layer_sizes[1:]:
        model.add(keras.layers.Dense(size, activation='sigmoid'))
    
    return model


def load_model_from_file(filepath):
    """Load a trained Keras model from file."""
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required")
    return keras.models.load_model(filepath)


def extract_weights_biases(model):
    """
    Extract weights and biases from a Keras model.
    
    Args:
        model: Keras model
    
    Returns:
        List of (weights, biases) tuples for each layer
        weights: numpy array of shape (num_inputs, num_neurons)
        biases: numpy array of shape (num_neurons,)
    """
    layer_params = []
    for layer in model.layers:
        weights, biases = layer.get_weights()
        layer_params.append((weights, biases))
    return layer_params


def float_to_fixed(value, frac_bits=16):
    """
    Convert float to fixed-point integer representation.
    
    Args:
        value: Float value to convert
        frac_bits: Number of fractional bits
    
    Returns:
        Integer representation
    """
    return int(value * (2 ** frac_bits))


def generate_weight_loading_vhdl(layer_params, data_width=32, frac_bits=16):
    """
    Generate VHDL code for weight loading sequence.
    
    Args:
        layer_params: List of (weights, biases) tuples
        data_width: VHDL data width
        frac_bits: Number of fractional bits
    
    Returns:
        String containing VHDL weight loading code
    """
    vhdl_code = []
    vhdl_code.append("    -- ========================================================================")
    vhdl_code.append("    -- Weight Loading Sequence")
    vhdl_code.append("    -- ========================================================================")
    vhdl_code.append("")
    
    for layer_idx, (weights, biases) in enumerate(layer_params):
        num_neurons, num_inputs = weights.shape[1], weights.shape[0]
        
        vhdl_code.append(f"    -- Layer {layer_idx}: {num_inputs} inputs -> {num_neurons} neurons")
        vhdl_code.append("")
        
        for neuron_idx in range(num_neurons):
            vhdl_code.append(f"    -- Neuron {neuron_idx}")
            
            # Load weights
            for weight_idx in range(num_inputs):
                weight_val = weights[weight_idx, neuron_idx]
                fixed_val = float_to_fixed(weight_val, frac_bits)
                
                # Convert to std_logic_vector representation
                if fixed_val < 0:
                    # Two's complement for negative numbers
                    slv_val = (1 << data_width) + fixed_val
                else:
                    slv_val = fixed_val
                
                vhdl_code.append(f"    layer_{layer_idx}_select <= {neuron_idx};")
                vhdl_code.append(f"    weight_index <= {weight_idx};")
                vhdl_code.append(f'    weight_data <= std_logic_vector(to_signed({fixed_val}, {data_width}));')
                vhdl_code.append(f"    load_enable <= '1';")
                vhdl_code.append(f"    wait until rising_edge(clk);")
                vhdl_code.append("")
            
            # Load bias
            bias_val = biases[neuron_idx]
            fixed_bias = float_to_fixed(bias_val, frac_bits)
            
            vhdl_code.append(f"    -- Bias for neuron {neuron_idx}")
            vhdl_code.append(f"    layer_{layer_idx}_select <= {neuron_idx};")
            vhdl_code.append(f"    weight_index <= {num_inputs};  -- Bias index")
            vhdl_code.append(f'    weight_data <= std_logic_vector(to_signed({fixed_bias}, {data_width}));')
            vhdl_code.append(f"    load_enable <= '1';")
            vhdl_code.append(f"    wait until rising_edge(clk);")
            vhdl_code.append("")
        
        vhdl_code.append("")
    
    vhdl_code.append("    load_enable <= '0';")
    vhdl_code.append("    wait until rising_edge(clk);")
    vhdl_code.append("")
    
    return "\n".join(vhdl_code)


def generate_test_vectors(model, num_tests=10, input_dim=4):
    """
    Generate random test vectors and compute expected outputs.
    
    Args:
        model: Keras model
        num_tests: Number of test cases
        input_dim: Input dimension
    
    Returns:
        (inputs, outputs) tuple of numpy arrays
    """
    # Generate random inputs in range [-1, 1]
    inputs = np.random.uniform(-1, 1, (num_tests, input_dim))
    
    # Compute expected outputs
    outputs = model.predict(inputs, verbose=0)
    
    return inputs, outputs


def generate_neural_network_testbench(model_name, layer_params, test_inputs, test_outputs, 
                                     config, output_file="testbench/neural_network_tb.vhd"):
    """
    Generate complete VHDL testbench for neural network.
    
    Args:
        model_name: Name of the model
        layer_params: List of (weights, biases) tuples
        test_inputs: Test input vectors
        test_outputs: Expected output vectors
        config: SigmoidConfig object
        output_file: Output file path
    """
    
    num_layers = len(layer_params)
    num_inputs = layer_params[0][0].shape[0]
    layer_sizes = [params[0].shape[1] for params in layer_params]
    num_outputs = layer_sizes[-1]
    
    # Calculate dynamic tolerance based on LUT step size
    input_step = (config.input_range[1] - config.input_range[0]) / config.lut_size
    max_lut_step = 0.25 * input_step
    tolerance = max_lut_step * num_layers  # Looser for multi-layer
    
    # Generate weight loading code
    weight_loading_vhdl = []
    weight_loading_vhdl.append("    load_mode <= '1';")
    weight_loading_vhdl.append("    wait until rising_edge(clk);")
    weight_loading_vhdl.append("")
    
    for layer_idx, (weights, biases) in enumerate(layer_params):
        num_neurons, num_inputs_layer = weights.shape[1], weights.shape[0]
        
        weight_loading_vhdl.append(f"    -- Layer {layer_idx}: {num_inputs_layer} inputs -> {num_neurons} neurons")
        weight_loading_vhdl.append(f"    layer_select <= {layer_idx};")
        weight_loading_vhdl.append("")
        
        for neuron_idx in range(num_neurons):
            weight_loading_vhdl.append(f"    -- Neuron {neuron_idx}")
            weight_loading_vhdl.append(f"    neuron_select <= {neuron_idx};")
            
            # Load weights
            for weight_idx in range(num_inputs_layer):
                weight_val = weights[weight_idx, neuron_idx]
                fixed_val = float_to_fixed(weight_val, config.input_frac_width)
                
                weight_loading_vhdl.append(f"    weight_index <= {weight_idx};")
                weight_loading_vhdl.append(f'    weight_data <= std_logic_vector(to_signed({fixed_val}, {config.input_width}));')
                weight_loading_vhdl.append(f"    wait until rising_edge(clk);")
            
            # Load bias
            bias_val = biases[neuron_idx]
            fixed_bias = float_to_fixed(bias_val, config.input_frac_width)
            
            weight_loading_vhdl.append(f"    weight_index <= {num_inputs_layer};  -- Bias")
            weight_loading_vhdl.append(f'    weight_data <= std_logic_vector(to_signed({fixed_bias}, {config.input_width}));')
            weight_loading_vhdl.append(f"    wait until rising_edge(clk);")
            weight_loading_vhdl.append("")
    
    weight_loading_vhdl.append("    load_mode <= '0';")
    weight_loading_vhdl.append("    wait until rising_edge(clk);")
    weight_loading_vhdl.append("    wait until rising_edge(clk);")
    
    # Generate test vector application code
    test_vector_vhdl = []
    for test_idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
        test_vector_vhdl.append(f"    -- Test {test_idx}")
        test_vector_vhdl.append(f"    report \"Test {test_idx}:\";")
        
        # Set inputs
        for input_idx, input_val in enumerate(test_input):
            fixed_val = float_to_fixed(input_val, config.input_frac_width)
            test_vector_vhdl.append(f"    inputs({input_idx}) <= std_logic_vector(to_signed({fixed_val}, {config.input_width}));")
        
        # Wait for computation (num_layers + 3 cycles for pipeline)
        wait_cycles = num_layers + 3
        for _ in range(wait_cycles):
            test_vector_vhdl.append("    wait until rising_edge(clk);")
        
        # Compare results
        test_vector_vhdl.append("    total_tests := total_tests + 1;")
        test_vector_vhdl.append("")
        
        for output_idx, expected_val in enumerate(expected_output):
            test_vector_vhdl.append(f"    output_real := to_real(to_sfixed(outputs({output_idx}), output_fixed));")
            test_vector_vhdl.append(f"    report \"  Output[{output_idx}] = \" & real'image(output_real) & \" (Expected: {expected_val:.6f})\";")
            test_vector_vhdl.append(f"    if abs(output_real - {expected_val:.6f}) < TOLERANCE_C then")
            test_vector_vhdl.append("        pass_count := pass_count + 1;")
            test_vector_vhdl.append(f'        report "  Output[{output_idx}] PASS";')
            test_vector_vhdl.append("    else")
            test_vector_vhdl.append("        fail_count := fail_count + 1;")
            test_vector_vhdl.append(f'        report "  Output[{output_idx}] FAIL";')
            test_vector_vhdl.append("    end if;")
        
        test_vector_vhdl.append("")
    
    # Complete VHDL testbench
    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity neural_network_tb is
end entity neural_network_tb;

architecture testbench of neural_network_tb is
    -- Configuration
    constant NUM_INPUTS : integer := {num_inputs};
    constant NUM_LAYERS : integer := {num_layers};
    constant LAYER_SIZE_0 : integer := {layer_sizes[0] if num_layers >= 1 else 1};
    constant LAYER_SIZE_1 : integer := {layer_sizes[1] if num_layers >= 2 else 1};
    constant LAYER_SIZE_2 : integer := {layer_sizes[2] if num_layers >= 3 else 1};
    constant LAYER_SIZE_3 : integer := {layer_sizes[3] if num_layers >= 4 else 1};
    constant NUM_OUTPUTS : integer := {num_outputs};
    constant DATA_WIDTH : integer := {config.input_width};
    constant TOLERANCE_C : real := {tolerance:.6f};
    
    -- Clock
    constant CLK_PERIOD : time := 10 ns;
    signal clk : std_logic := '0';
    signal stop_clock : boolean := false;
    
    -- DUT signals
    signal rst : std_logic := '0';
    signal load_mode : std_logic := '0';
    signal layer_select : integer range 0 to NUM_LAYERS - 1 := 0;
    signal neuron_select : integer range 0 to 15 := 0;
    signal weight_index : integer range 0 to 15 := 0;
    signal weight_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal inputs : std_logic_bus_array(NUM_INPUTS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    signal outputs : std_logic_bus_array(0 to 15)(DATA_WIDTH - 1 downto 0);
    signal output_valid : std_logic;
    signal ready : std_logic;
    
    -- Test signals
    signal output_fixed : sfixed((DATA_WIDTH + 1) / 2 - 1 downto -(DATA_WIDTH / 2));

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
    end process;
    
    -- ========================================================================
    -- DUT Instantiation
    -- ========================================================================
    dut: entity work.neural_network
        generic map(
            num_inputs => NUM_INPUTS,
            num_layers => NUM_LAYERS,
            layer_size_0 => LAYER_SIZE_0,
            layer_size_1 => LAYER_SIZE_1,
            layer_size_2 => LAYER_SIZE_2,
            layer_size_3 => LAYER_SIZE_3,
            data_width => DATA_WIDTH,
            use_sigmoid => true
        )
        port map(
            clk => clk,
            rst => rst,
            load_mode => load_mode,
            layer_select => layer_select,
            neuron_select => neuron_select,
            weight_index => weight_index,
            weight_data => weight_data,
            inputs_i => inputs,
            outputs_o => outputs,
            output_valid_o => output_valid,
            ready_o => ready
        );
    
    -- ========================================================================
    -- Test Process
    -- ========================================================================
    test_proc: process
        variable pass_count : integer := 0;
        variable fail_count : integer := 0;
        variable total_tests : integer := 0;
        variable output_real : real;
    begin
        report "========================================";
        report "Neural Network Testbench";
        report "Model: {model_name}";
        report "Layers: {' -> '.join(map(str, layer_sizes))}";
        report "Tolerance: " & real'image(TOLERANCE_C);
        report "========================================";
        
        -- Reset
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD;
        
        -- ====================================================================
        -- Weight Loading
        -- ====================================================================
        report "Loading weights...";
{chr(10).join('        ' + line for line in weight_loading_vhdl)}
        
        report "Weights loaded. Starting inference tests...";
        wait for CLK_PERIOD * 2;
        
        -- ====================================================================
        -- Test Vectors
        -- ====================================================================
{chr(10).join('        ' + line for line in test_vector_vhdl)}
        
        -- ====================================================================
        -- Test Summary
        -- ====================================================================
        report "========================================";
        report "Test Summary:";
        report "  Total: " & integer'image(total_tests);
        report "  Passed: " & integer'image(pass_count);
        report "  Failed: " & integer'image(fail_count);
        report "========================================";
        
        if fail_count = 0 then
            report "ALL TESTS PASSED!" severity note;
        else
            report "SOME TESTS FAILED!" severity warning;
        end if;
        
        stop_clock <= true;
        wait;
    end process;

end architecture testbench;
"""
    
    # Write to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)
    
    print(f"Neural Network Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Inputs: {num_inputs}")
    print(f"  Layers: {num_layers}")
    print(f"  Layer sizes: {layer_sizes}")
    print(f"  Outputs: {num_outputs}")
    print(f"  Total test cases: {len(test_inputs)}")
    print(f"  Tolerance: {tolerance:.6f}")
    print(f"\n✓ Generated testbench: {output_file}")



def main():
    parser = argparse.ArgumentParser(
        description='Generate VHDL testbench for neural network using Keras model'
    )
    parser.add_argument('--model-file', type=str,
                        help='Path to saved Keras model file')
    parser.add_argument('--create-model', action='store_true',
                        help='Create a simple model for testing')
    parser.add_argument('--layers', type=str, default='4,3,2',
                        help='Layer sizes (comma-separated), e.g., "4,3,2" for 4->3->2 network')
    parser.add_argument('--input-dim', type=int, default=4,
                        help='Input dimension (default: 4)')
    parser.add_argument('--num-tests', type=int, default=10,
                        help='Number of test cases (default: 10)')
    parser.add_argument('--data-width', type=int, default=32,
                        help='VHDL data width (default: 32)')
    parser.add_argument('--frac-bits', type=int, default=16,
                        help='Fractional bits (default: 16)')
    parser.add_argument('--output', type=str, default='testbench/neural_network_tb.vhd',
                        help='Output testbench file')
    
    args = parser.parse_args()
    
    if not KERAS_AVAILABLE:
        print("ERROR: TensorFlow/Keras is not installed.")
        print("Install with: pip install tensorflow")
        return 1
    
    # Create or load model
    if args.model_file:
        print(f"Loading model from {args.model_file}...")
        model = load_model_from_file(args.model_file)
    elif args.create_model:
        layer_sizes = [int(x) for x in args.layers.split(',')]
        print(f"Creating simple model: {args.input_dim} -> {' -> '.join(map(str, layer_sizes))}")
        model = create_simple_model(layer_sizes, args.input_dim)
        
        # Train on dummy data for demonstration
        X_train = np.random.uniform(-1, 1, (100, args.input_dim))
        y_train = np.random.uniform(0, 1, (100, layer_sizes[-1]))
        model.compile(optimizer='adam', loss='mse')
        print("Training model (this is just for demo)...")
        model.fit(X_train, y_train, epochs=10, verbose=0)
    else:
        print("ERROR: Either --model-file or --create-model must be specified")
        return 1
    
    # Extract weights and biases
    print("\nExtracting weights and biases...")
    layer_params = extract_weights_biases(model)
    
    # Generate test vectors
    print(f"Generating {args.num_tests} test vectors...")
    test_inputs, test_outputs = generate_test_vectors(model, args.num_tests, args.input_dim)
    
    # Create config
    config = SigmoidConfig(
        input_width=args.data_width,
        input_frac_width=args.frac_bits,
        output_width=args.data_width
    )
    
    # Generate testbench
    print(f"\nGenerating testbench: {args.output}")
    generate_neural_network_testbench(
        "test_network",
        layer_params,
        test_inputs,
        test_outputs,
        config,
        args.output
    )
    
    print("\n✓ Testbench generation complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())

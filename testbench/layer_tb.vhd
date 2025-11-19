library ieee;
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
    constant DATA_WIDTH_C : integer := 16;
    constant USE_SIGMOID_C : boolean := true;

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
    begin
        report "========================================";
        report "Starting Layer Testbench";
        report "Data Width: " & integer'image(DATA_WIDTH_C);
        report "Activation: Sigmoid";
        report "========================================";

        -- ====================================================================
        -- TEST 1: Single Layer (4 inputs -> 2 outputs)
        -- ====================================================================
        report "========================================";
        report "TEST 1: Single Layer (4 inputs -> 2 outputs)";
        report "========================================";

        -- Initialize inputs for Test 1
        -- Inputs: [0.5, 1.0, -0.5, 0.25]
        input_fixed := to_sfixed(0.5, input_fixed'high, input_fixed'low);
        inputs_1(0) <= to_slv(input_fixed);

        input_fixed := to_sfixed(1.0, input_fixed'high, input_fixed'low);
        inputs_1(1) <= to_slv(input_fixed);

        input_fixed := to_sfixed(-0.5, input_fixed'high, input_fixed'low);
        inputs_1(2) <= to_slv(input_fixed);

        input_fixed := to_sfixed(0.25, input_fixed'high, input_fixed'low);
        inputs_1(3) <= to_slv(input_fixed);

        -- Set weights for Test 1
        -- Neuron 0: weights=[1.0, 0.5, -0.5, 0.75], bias=0.5
        -- Neuron 1: weights=[0.5, 1.0, 0.25, -0.5], bias=-0.25

        -- Neuron 0 weights
        weight_fixed := to_sfixed(1.0, weight_fixed'high, weight_fixed'low);
        weights_1(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);
        weights_1(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.5, weight_fixed'high, weight_fixed'low);
        weights_1(0)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.75, weight_fixed'high, weight_fixed'low);
        weights_1(0)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);  -- bias
        weights_1(0)(4) <= to_slv(weight_fixed);

        -- Neuron 1 weights
        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);
        weights_1(1)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(1.0, weight_fixed'high, weight_fixed'low);
        weights_1(1)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.25, weight_fixed'high, weight_fixed'low);
        weights_1(1)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.5, weight_fixed'high, weight_fixed'low);
        weights_1(1)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.25, weight_fixed'high, weight_fixed'low);  -- bias
        weights_1(1)(4) <= to_slv(weight_fixed);

        report "Inputs set: [0.5, 1.0, -0.5, 0.25]";
        report "Weights configured for 2 neurons";
        report "Waiting 2 clock cycles for output...";

        -- Wait 2 clock cycles for pipeline
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 4;  -- Wait a bit into the cycle for signals to settle

        report "Test 1 Output:";
        report "  Output[0] = " & real'image(output_1_real(0));
        report "  Output[1] = " & real'image(output_1_real(1));

        wait for CLK_PERIOD * 2;

        -- ====================================================================
        -- TEST 2: Two Layers (4 -> 2 -> 1)
        -- ====================================================================
        report "========================================";
        report "TEST 2: Two Layers (4 -> 2 -> 1)";
        report "========================================";

        -- Initialize inputs for Test 2
        -- Inputs: [1.0, 0.5, -1.0, 0.75]
        input_fixed := to_sfixed(1.0, input_fixed'high, input_fixed'low);
        inputs_2_layer1(0) <= to_slv(input_fixed);

        input_fixed := to_sfixed(0.5, input_fixed'high, input_fixed'low);
        inputs_2_layer1(1) <= to_slv(input_fixed);

        input_fixed := to_sfixed(-1.0, input_fixed'high, input_fixed'low);
        inputs_2_layer1(2) <= to_slv(input_fixed);

        input_fixed := to_sfixed(0.75, input_fixed'high, input_fixed'low);
        inputs_2_layer1(3) <= to_slv(input_fixed);

        -- Set weights for Layer 1 (4 inputs -> 2 outputs)
        -- Neuron 0: weights=[0.8, 0.3, -0.4, 0.6], bias=0.2
        -- Neuron 1: weights=[0.4, 0.9, 0.1, -0.3], bias=-0.1

        -- Layer 1, Neuron 0 weights
        weight_fixed := to_sfixed(0.8, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.3, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.4, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.6, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(0)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.2, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer1(0)(4) <= to_slv(weight_fixed);

        -- Layer 1, Neuron 1 weights
        weight_fixed := to_sfixed(0.4, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.9, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.1, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(2) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.3, weight_fixed'high, weight_fixed'low);
        weights_2_layer1(1)(3) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(-0.1, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer1(1)(4) <= to_slv(weight_fixed);

        -- Set weights for Layer 2 (2 inputs -> 1 output)
        -- Neuron 0: weights=[1.2, 0.7], bias=0.3

        weight_fixed := to_sfixed(1.2, weight_fixed'high, weight_fixed'low);
        weights_2_layer2(0)(0) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.7, weight_fixed'high, weight_fixed'low);
        weights_2_layer2(0)(1) <= to_slv(weight_fixed);

        weight_fixed := to_sfixed(0.3, weight_fixed'high, weight_fixed'low);  -- bias
        weights_2_layer2(0)(2) <= to_slv(weight_fixed);

        report "Inputs set: [1.0, 0.5, -1.0, 0.75]";
        report "Weights configured for Layer 1 (4->2) and Layer 2 (2->1)";
        report "Waiting 4 clock cycles for final output...";

        -- Wait 4 clock cycles for both layers
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        report "After 2 cycles - Layer 1 output:";
        report "  Layer1 Output[0] = " & real'image(output_2_layer1_real(0));
        report "  Layer1 Output[1] = " & real'image(output_2_layer1_real(1));

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 4;  -- Wait a bit into the cycle for signals to settle

        report "After 4 cycles - Final output:";
        report "  Layer2 Output[0] = " & real'image(output_2_layer2_real(0));

        wait for CLK_PERIOD * 2;

        -- ====================================================================
        -- Test Complete
        -- ====================================================================
        report "========================================";
        report "All Tests Complete!";
        report "========================================";

        stop_clock <= true;
        wait;
    end process test_proc;

end architecture testbench;

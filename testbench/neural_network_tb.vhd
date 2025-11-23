library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity neural_network_tb is
end entity neural_network_tb;

architecture testbench of neural_network_tb is
    -- Configuration
    constant NUM_INPUTS : integer := 4;
    constant NUM_OUTPUTS : integer := 2;
    constant DATA_WIDTH : integer := 32;
    constant TOLERANCE_C : real := 0.046875;
    
    -- Layer configuration
    constant LAYER_SIZES : layer_config_array(0 to 2) := (4, 3, 2);
    
    -- Clock
    constant CLK_PERIOD : time := 10 ns;
    signal clk : std_logic := '0';
    signal stop_clock : boolean := false;
    
    -- DUT signals
    signal rst : std_logic := '0';
    signal load_mode : std_logic := '0';
    signal layer_select : integer range 0 to 15 := 0;
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
            layer_sizes => LAYER_SIZES,
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
        report "Model: test_network";
        report "Layers: 4 -> 3 -> 2";
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
            load_mode <= '1';
            wait until rising_edge(clk);
        
            -- Layer 0: 4 inputs -> 4 neurons
            layer_select <= 0;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-28009, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(29199, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(11891, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(11297, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-448, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-38573, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-41537, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-43780, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(23940, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-1543, 32));
            wait until rising_edge(clk);
        
            -- Neuron 2
            neuron_select <= 2;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-35501, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(39059, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-39314, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(-47212, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(235, 32));
            wait until rising_edge(clk);
        
            -- Neuron 3
            neuron_select <= 3;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(16486, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(20220, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-27618, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(33252, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-1599, 32));
            wait until rising_edge(clk);
        
            -- Layer 1: 4 inputs -> 3 neurons
            layer_select <= 1;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-31162, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(10092, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-33363, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(51176, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-1457, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-58792, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-26555, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-52474, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(-50712, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(1609, 32));
            wait until rising_edge(clk);
        
            -- Neuron 2
            neuron_select <= 2;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-21764, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-3949, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(21683, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(22129, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(360, 32));
            wait until rising_edge(clk);
        
            -- Layer 2: 3 inputs -> 2 neurons
            layer_select <= 2;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-40696, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(68969, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-4562, 32));
            wait until rising_edge(clk);
            weight_index <= 3;  -- Bias
            weight_data <= std_logic_vector(to_signed(1656, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(57295, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-9884, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-16408, 32));
            wait until rising_edge(clk);
            weight_index <= 3;  -- Bias
            weight_data <= std_logic_vector(to_signed(-857, 32));
            wait until rising_edge(clk);
        
            load_mode <= '0';
            wait until rising_edge(clk);
            wait until rising_edge(clk);
        
        report "Weights loaded. Starting inference tests...";
        wait for CLK_PERIOD * 2;
        
        -- ====================================================================
        -- Test Vectors
        -- ====================================================================
            -- Test 0
            report "Test 0:";
            inputs(0) <= std_logic_vector(to_signed(23263, 32));
            inputs(1) <= std_logic_vector(to_signed(64689, 32));
            inputs(2) <= std_logic_vector(to_signed(38031, 32));
            inputs(3) <= std_logic_vector(to_signed(-60845, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.481963)";
            if abs(output_real - 0.481963) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.549240)";
            if abs(output_real - 0.549240) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 1
            report "Test 1:";
            inputs(0) <= std_logic_vector(to_signed(-16773, 32));
            inputs(1) <= std_logic_vector(to_signed(-59810, 32));
            inputs(2) <= std_logic_vector(to_signed(3688, 32));
            inputs(3) <= std_logic_vector(to_signed(14557, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.476659)";
            if abs(output_real - 0.476659) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.566324)";
            if abs(output_real - 0.566324) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 2
            report "Test 2:";
            inputs(0) <= std_logic_vector(to_signed(60526, 32));
            inputs(1) <= std_logic_vector(to_signed(11385, 32));
            inputs(2) <= std_logic_vector(to_signed(-116, 32));
            inputs(3) <= std_logic_vector(to_signed(21788, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.472685)";
            if abs(output_real - 0.472685) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.570215)";
            if abs(output_real - 0.570215) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 3
            report "Test 3:";
            inputs(0) <= std_logic_vector(to_signed(-50368, 32));
            inputs(1) <= std_logic_vector(to_signed(21632, 32));
            inputs(2) <= std_logic_vector(to_signed(-44329, 32));
            inputs(3) <= std_logic_vector(to_signed(62833, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.455945)";
            if abs(output_real - 0.455945) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.567001)";
            if abs(output_real - 0.567001) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 4
            report "Test 4:";
            inputs(0) <= std_logic_vector(to_signed(8451, 32));
            inputs(1) <= std_logic_vector(to_signed(54007, 32));
            inputs(2) <= std_logic_vector(to_signed(54476, 32));
            inputs(3) <= std_logic_vector(to_signed(-14012, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.479982)";
            if abs(output_real - 0.479982) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.554837)";
            if abs(output_real - 0.554837) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
        
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

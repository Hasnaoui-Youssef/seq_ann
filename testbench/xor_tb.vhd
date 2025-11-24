library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity xor_tb is
end entity xor_tb;

architecture testbench of xor_tb is
    -- Configuration
    constant NUM_INPUTS : integer := 2;
    constant NUM_OUTPUTS : integer := 1;
    constant DATA_WIDTH : integer := 32;
    constant TOLERANCE_C : real := 0.1;
    
    -- Layer configuration
    constant LAYER_SIZES : layer_config_array(0 to 1) := (3, 1);
    
    -- Clock
    constant CLK_PERIOD : time := 10 ns;
    signal clk : std_logic := '0';
    signal stop_clock : boolean := false;
    
    -- DUT signals
    signal rst : std_logic := '0';
    signal load_mode : std_logic := '0';
    signal layer_select : integer range 0 to LAYER_SIZES'high := 0;
    signal neuron_select : integer := 0;
    signal weight_index : integer := 0;
    signal weight_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal inputs : std_logic_bus_array(NUM_INPUTS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    signal outputs : std_logic_bus_array(0 to LAYER_SIZES(LAYER_SIZES'high) - 1)(DATA_WIDTH - 1 downto 0);
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
            use_sigmoid => true
        )
        port map(
            clk => clk,
            rst => rst,
            load_mode => load_mode,
            neuron_select => neuron_select,
            layer_select => layer_select,
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
        report "Layers: 3 -> 1";
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
        
            -- Layer 0: 2 inputs -> 3 neurons
            layer_select <= 0;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-1533, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(38171, 32));
            wait until rising_edge(clk);
            weight_index <= 2;  -- Bias
            weight_data <= std_logic_vector(to_signed(-36900, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(363017, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-351147, 32));
            wait until rising_edge(clk);
            weight_index <= 2;  -- Bias
            weight_data <= std_logic_vector(to_signed(182737, 32));
            wait until rising_edge(clk);
        
            -- Neuron 2
            neuron_select <= 2;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(321004, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-340674, 32));
            wait until rising_edge(clk);
            weight_index <= 2;  -- Bias
            weight_data <= std_logic_vector(to_signed(-173674, 32));
            wait until rising_edge(clk);
        
            -- Layer 1: 3 inputs -> 1 neurons
            layer_select <= 1;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(76395, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-489043, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(523118, 32));
            wait until rising_edge(clk);
            weight_index <= 3;  -- Bias
            weight_data <= std_logic_vector(to_signed(193707, 32));
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
            inputs(0) <= std_logic_vector(to_signed(0, 32));
            inputs(1) <= std_logic_vector(to_signed(0, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.000000)";
            if abs(output_real - 0.000000) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
        
            -- Test 1
            report "Test 1:";
            inputs(0) <= std_logic_vector(to_signed(0, 32));
            inputs(1) <= std_logic_vector(to_signed(65536, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 1.000000)";
            if abs(output_real - 1.000000) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
        
            -- Test 2
            report "Test 2:";
            inputs(0) <= std_logic_vector(to_signed(65536, 32));
            inputs(1) <= std_logic_vector(to_signed(0, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 1.000000)";
            if abs(output_real - 1.000000) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
        
            -- Test 3
            report "Test 3:";
            inputs(0) <= std_logic_vector(to_signed(65536, 32));
            inputs(1) <= std_logic_vector(to_signed(65536, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.000000)";
            if abs(output_real - 0.000000) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
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

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
    constant NUM_LAYERS : integer := 3;
    constant LAYER_SIZE_0 : integer := 4;
    constant LAYER_SIZE_1 : integer := 3;
    constant LAYER_SIZE_2 : integer := 2;
    constant LAYER_SIZE_3 : integer := 1;
    constant NUM_OUTPUTS : integer := 2;
    constant DATA_WIDTH : integer := 32;
    constant TOLERANCE_C : real := 0.046875;
    
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
            weight_data <= std_logic_vector(to_signed(33338, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(42521, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-26765, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(-21589, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(350, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-24616, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(50061, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(32534, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(27983, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(1818, 32));
            wait until rising_edge(clk);
        
            -- Neuron 2
            neuron_select <= 2;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-37709, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-41983, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-48308, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(9050, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(2419, 32));
            wait until rising_edge(clk);
        
            -- Neuron 3
            neuron_select <= 3;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-23418, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-10738, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-47814, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(-4733, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-2296, 32));
            wait until rising_edge(clk);
        
            -- Layer 1: 4 inputs -> 3 neurons
            layer_select <= 1;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(37149, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-24173, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-23556, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(-6189, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-2373, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-46322, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(19847, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-62386, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(19550, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(-2399, 32));
            wait until rising_edge(clk);
        
            -- Neuron 2
            neuron_select <= 2;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(18302, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(29332, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-8344, 32));
            wait until rising_edge(clk);
            weight_index <= 3;
            weight_data <= std_logic_vector(to_signed(4469, 32));
            wait until rising_edge(clk);
            weight_index <= 4;  -- Bias
            weight_data <= std_logic_vector(to_signed(1796, 32));
            wait until rising_edge(clk);
        
            -- Layer 2: 3 inputs -> 2 neurons
            layer_select <= 2;
        
            -- Neuron 0
            neuron_select <= 0;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(12699, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-10392, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(-7750, 32));
            wait until rising_edge(clk);
            weight_index <= 3;  -- Bias
            weight_data <= std_logic_vector(to_signed(1330, 32));
            wait until rising_edge(clk);
        
            -- Neuron 1
            neuron_select <= 1;
            weight_index <= 0;
            weight_data <= std_logic_vector(to_signed(-60078, 32));
            wait until rising_edge(clk);
            weight_index <= 1;
            weight_data <= std_logic_vector(to_signed(-48033, 32));
            wait until rising_edge(clk);
            weight_index <= 2;
            weight_data <= std_logic_vector(to_signed(8778, 32));
            wait until rising_edge(clk);
            weight_index <= 3;  -- Bias
            weight_data <= std_logic_vector(to_signed(2414, 32));
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
            inputs(0) <= std_logic_vector(to_signed(-36794, 32));
            inputs(1) <= std_logic_vector(to_signed(887, 32));
            inputs(2) <= std_logic_vector(to_signed(-38720, 32));
            inputs(3) <= std_logic_vector(to_signed(57546, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.494353)";
            if abs(output_real - 0.494353) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.372880)";
            if abs(output_real - 0.372880) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 1
            report "Test 1:";
            inputs(0) <= std_logic_vector(to_signed(28612, 32));
            inputs(1) <= std_logic_vector(to_signed(55122, 32));
            inputs(2) <= std_logic_vector(to_signed(58919, 32));
            inputs(3) <= std_logic_vector(to_signed(-52599, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.495039)";
            if abs(output_real - 0.495039) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.346256)";
            if abs(output_real - 0.346256) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 2
            report "Test 2:";
            inputs(0) <= std_logic_vector(to_signed(63470, 32));
            inputs(1) <= std_logic_vector(to_signed(-4067, 32));
            inputs(2) <= std_logic_vector(to_signed(58682, 32));
            inputs(3) <= std_logic_vector(to_signed(57359, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.493645)";
            if abs(output_real - 0.493645) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.350692)";
            if abs(output_real - 0.350692) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 3
            report "Test 3:";
            inputs(0) <= std_logic_vector(to_signed(-41301, 32));
            inputs(1) <= std_logic_vector(to_signed(-2361, 32));
            inputs(2) <= std_logic_vector(to_signed(-37522, 32));
            inputs(3) <= std_logic_vector(to_signed(27366, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.495007)";
            if abs(output_real - 0.495007) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.372194)";
            if abs(output_real - 0.372194) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[1] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[1] FAIL";
            end if;
        
            -- Test 4
            report "Test 4:";
            inputs(0) <= std_logic_vector(to_signed(-27893, 32));
            inputs(1) <= std_logic_vector(to_signed(-40037, 32));
            inputs(2) <= std_logic_vector(to_signed(14058, 32));
            inputs(3) <= std_logic_vector(to_signed(50717, 32));
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            wait until rising_edge(clk);
            total_tests := total_tests + 1;
        
            output_real := to_real(to_sfixed(outputs(0), output_fixed));
            report "  Output[0] = " & real'image(output_real) & " (Expected: 0.493442)";
            if abs(output_real - 0.493442) < TOLERANCE_C then
                pass_count := pass_count + 1;
                report "  Output[0] PASS";
            else
                fail_count := fail_count + 1;
                report "  Output[0] FAIL";
            end if;
            output_real := to_real(to_sfixed(outputs(1), output_fixed));
            report "  Output[1] = " & real'image(output_real) & " (Expected: 0.369768)";
            if abs(output_real - 0.369768) < TOLERANCE_C then
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

library ieee;
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
    constant TOLERANCE_C : real := 0.031250;
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
        -- Weights: [1.0, 0.5, -0.5, 0.75, 0.5]
        -- w0
        weight_fixed := to_sfixed(1.0, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 0; load_enable <= '1'; wait until rising_edge(clk);
        -- w1
        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 1; load_enable <= '1'; wait until rising_edge(clk);
        -- w2
        weight_fixed := to_sfixed(-0.5, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 2; load_enable <= '1'; wait until rising_edge(clk);
        -- w3
        weight_fixed := to_sfixed(0.75, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 3; load_enable <= '1'; wait until rising_edge(clk);
        -- bias
        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 4; load_enable <= '1'; wait until rising_edge(clk);

        -- Neuron 1
        neuron_select <= 1;
        -- Weights: [0.5, 1.0, 0.25, -0.5, -0.25]
        -- w0
        weight_fixed := to_sfixed(0.5, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 0; load_enable <= '1'; wait until rising_edge(clk);
        -- w1
        weight_fixed := to_sfixed(1.0, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 1; load_enable <= '1'; wait until rising_edge(clk);
        -- w2
        weight_fixed := to_sfixed(0.25, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 2; load_enable <= '1'; wait until rising_edge(clk);
        -- w3
        weight_fixed := to_sfixed(-0.5, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 3; load_enable <= '1'; wait until rising_edge(clk);
        -- bias
        weight_fixed := to_sfixed(-0.25, weight_fixed'high, weight_fixed'low);
        weight_data <= to_slv(weight_fixed); weight_index <= 4; load_enable <= '1'; wait until rising_edge(clk);

        load_enable <= '0';
        wait until rising_edge(clk);
        report "Weights loaded.";

        -- 2. Set Inputs
        -- Inputs: [0.5, 1.0, -0.5, 0.25]
        input_fixed := to_sfixed(0.5, input_fixed'high, input_fixed'low);
        inputs_s(0) <= to_slv(input_fixed);
        input_fixed := to_sfixed(1.0, input_fixed'high, input_fixed'low);
        inputs_s(1) <= to_slv(input_fixed);
        input_fixed := to_sfixed(-0.5, input_fixed'high, input_fixed'low);
        inputs_s(2) <= to_slv(input_fixed);
        input_fixed := to_sfixed(0.25, input_fixed'high, input_fixed'low);
        inputs_s(3) <= to_slv(input_fixed);

        -- 3. Wait for computation
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for CLK_PERIOD / 2;

        -- 4. Check Outputs
        -- Output 0
        output_real := to_real(to_sfixed(output_s(0), input_fixed));
        report "Output[0]: " & real'image(output_real) & " Expected: 0.874077";
        if abs(output_real - 0.874077) < TOLERANCE_C then
            pass_count := pass_count + 1; report "PASS";
        else
            fail_count := fail_count + 1; report "FAIL";
        end if;

        -- Output 1
        output_real := to_real(to_sfixed(output_s(1), input_fixed));
        report "Output[1]: " & real'image(output_real) & " Expected: 0.679179";
        if abs(output_real - 0.679179) < TOLERANCE_C then
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

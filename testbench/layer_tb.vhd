library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.pkg_layer.all;

entity layer_tb is
end entity layer_tb;

architecture testbench of layer_tb is
    -- Configuration
    constant NUM_INPUTS : integer := 4;
    constant NUM_OUTPUTS : integer := 2;
    constant NUM_WEIGHTS : integer := (NUM_INPUTS + 1) * NUM_OUTPUTS;
    constant USE_SIGMOID_C : boolean := true;
    constant TOLERANCE_C : real := 0.031250;
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

    -- Weight Bank Interface
    signal weight_load_en : std_logic := '0';
    signal weight_load_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_load_done : std_logic;
    signal weight_save_en : std_logic := '0';
    signal weight_save_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal weight_save_done : std_logic;
    signal weight_update_en : std_logic := '0';
    signal weight_learn_rate : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_update_done : std_logic;

    -- Gradients
    signal grads_s : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

    -- Weight values to load
    type weight_array_t is array (0 to NUM_WEIGHTS - 1) of real;
    constant WEIGHTS : weight_array_t := (1.0, 0.5, -0.5, 0.75, 0.5, 0.5, 1.0, 0.25, -0.5, -0.25);

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
            weight_load_en => weight_load_en,
            weight_load_data => weight_load_data,
            weight_load_done => weight_load_done,
            weight_save_en => weight_save_en,
            weight_save_data => weight_save_data,
            weight_save_done => weight_save_done,
            weight_update_en => weight_update_en,
            weight_learn_rate => weight_learn_rate,
            weight_update_done => weight_update_done,
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

        -- Load weights sequentially into weight bank
        report "Loading weights into weight bank...";
        for i in 0 to NUM_WEIGHTS - 1 loop
            weight_load_data <= to_slv(to_sfixed(WEIGHTS(i), INT_BITS - 1, -FRAC_BITS));
            weight_load_en <= '1';
            wait until rising_edge(clk);
            weight_load_en <= '0';
            wait until rising_edge(clk);
        end loop;
        report "Weights loaded.";

        -- Set inputs
        report "Setting inputs...";
        inputs_s(0) <= to_slv(to_sfixed(0.5, INT_BITS - 1, -FRAC_BITS));
        inputs_s(1) <= to_slv(to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS));
        inputs_s(2) <= to_slv(to_sfixed(-0.5, INT_BITS - 1, -FRAC_BITS));
        inputs_s(3) <= to_slv(to_sfixed(0.25, INT_BITS - 1, -FRAC_BITS));

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
        report "Output[0]: " & real'image(output_real) & " Expected: 0.874077";
        if abs(output_real - 0.874077) < TOLERANCE_C then
            pass_count := pass_count + 1; report "PASS";
        else
            fail_count := fail_count + 1; report "FAIL";
        end if;

        -- Output 1
        output_real := to_real(to_sfixed(outputs_s(1), INT_BITS - 1, -FRAC_BITS));
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

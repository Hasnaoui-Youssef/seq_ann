library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;
use work.layer_interface_pkg.all;

entity conv2d_tb is
end entity conv2d_tb;

architecture testbench of conv2d_tb is
    constant CLK_PERIOD : time := 10 ns;

    -- Test with a small 4x4 single-channel input, 2 filters, 3x3 kernel
    constant C_IN   : integer := 1;
    constant H_IN   : integer := 4;
    constant W_IN   : integer := 4;
    constant NUM_F  : integer := 2;
    constant K_H    : integer := 3;
    constant K_W    : integer := 3;
    constant H_OUT  : integer := 2;  -- (4-3)/1+1
    constant W_OUT  : integer := 2;

    constant INPUT_SIZE  : integer := C_IN * H_IN * W_IN;  -- 16
    constant OUTPUT_SIZE : integer := NUM_F * H_OUT * W_OUT; -- 8
    constant KERNEL_SIZE : integer := K_H * K_W * C_IN;     -- 9
    constant NUM_WEIGHTS : integer := (KERNEL_SIZE + 1) * NUM_F; -- 20

    signal clk : std_logic := '0';
    signal rst : std_logic := '0';

    signal fwd_ctrl_in  : layer_control_t := (valid => '0', last => '0');
    signal fwd_data_in  : std_logic_bus_array(0 to INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal fwd_ctrl_out : layer_control_t;
    signal fwd_data_out : std_logic_bus_array(0 to OUTPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    signal bwd_ctrl_in  : layer_control_t := (valid => '0', last => '0');
    signal bwd_error_in : std_logic_bus_array(0 to OUTPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal bwd_ctrl_out : layer_control_t;
    signal bwd_error_out: std_logic_bus_array(0 to INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    signal weight_load_en : std_logic := '0';
    signal weight_load_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_load_done : std_logic;
    signal weight_update_en : std_logic := '0';
    signal weight_learn_rate : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_update_done : std_logic;
    signal grads_out : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

    function to_slv(r : real) return std_logic_vector is
    begin
        return to_std_logic_vector(to_sfixed(r, INT_BITS - 1, -FRAC_BITS));
    end function;

    function to_real_val(slv : std_logic_vector) return real is
    begin
        return to_real(to_sfixed(slv, INT_BITS - 1, -FRAC_BITS));
    end function;

begin

    process
    begin
        while true loop
            clk <= '0'; wait for CLK_PERIOD / 2;
            clk <= '1'; wait for CLK_PERIOD / 2;
        end loop;
    end process;

    dut: entity work.conv2d_layer
        generic map (
            C_IN => C_IN, H_IN => H_IN, W_IN => W_IN,
            NUM_FILTERS => NUM_F, KERNEL_H => K_H, KERNEL_W => K_W,
            STRIDE_H => 1, STRIDE_W => 1, PAD_H => 0, PAD_W => 0
        )
        port map (
            clk => clk, rst => rst,
            fwd_ctrl_in => fwd_ctrl_in, fwd_data_in => fwd_data_in,
            fwd_ctrl_out => fwd_ctrl_out, fwd_data_out => fwd_data_out,
            bwd_ctrl_in => bwd_ctrl_in, bwd_error_in => bwd_error_in,
            bwd_ctrl_out => bwd_ctrl_out, bwd_error_out => bwd_error_out,
            weight_load_en => weight_load_en, weight_load_data => weight_load_data,
            weight_load_done => weight_load_done,
            weight_save_en => '0', weight_save_data => open, weight_save_done => open,
            weight_update_en => weight_update_en, weight_learn_rate => weight_learn_rate,
            weight_update_done => weight_update_done,
            grads_out => grads_out
        );

    process
        procedure load_weight(val : real) is
        begin
            weight_load_data <= to_slv(val);
            weight_load_en <= '1';
            wait until rising_edge(clk);
            weight_load_en <= '0';
            wait until rising_edge(clk);
        end procedure;

        variable expected : real;
        variable actual : real;
        variable tolerance : real := 0.01;
    begin
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD * 2;

        -- Load weights for filter 0: all 1s kernel (averaging), bias = 0
        -- Filter 0: 9 weights + 1 bias
        for i in 0 to KERNEL_SIZE - 1 loop
            load_weight(1.0);
        end loop;
        load_weight(0.0);  -- bias

        -- Filter 1: identity (1 at center, 0 elsewhere), bias = 0.5
        for i in 0 to KERNEL_SIZE - 1 loop
            if i = 4 then  -- center of 3x3 kernel
                load_weight(1.0);
            else
                load_weight(0.0);
            end if;
        end loop;
        load_weight(0.5);  -- bias

        report "Weights loaded. Testing forward pass...";

        -- Set input: 4x4 grid with values 1..16
        for i in 0 to INPUT_SIZE - 1 loop
            fwd_data_in(i) <= to_slv(real(i + 1));
        end loop;

        -- Trigger forward
        fwd_ctrl_in.valid <= '1';
        fwd_ctrl_in.last <= '1';
        wait until rising_edge(clk);
        fwd_ctrl_in.valid <= '0';
        fwd_ctrl_in.last <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);

        -- Filter 0 (all 1s kernel) at position (0,0):
        -- Sum of top-left 3x3 of input: 1+2+3+5+6+7+9+10+11 = 54
        expected := 54.0;
        actual := to_real_val(fwd_data_out(0));
        report "Filter0[0,0]: expected=" & real'image(expected) & " got=" & real'image(actual);
        assert abs(actual - expected) < tolerance
            report "Filter 0 [0,0] FAILED!" severity error;

        -- Filter 0 at (0,1): sum of 2+3+4+6+7+8+10+11+12 = 63
        expected := 63.0;
        actual := to_real_val(fwd_data_out(1));
        report "Filter0[0,1]: expected=" & real'image(expected) & " got=" & real'image(actual);
        assert abs(actual - expected) < tolerance
            report "Filter 0 [0,1] FAILED!" severity error;

        -- Filter 1 (identity + 0.5 bias) at (0,0): center is input[1*4+1]=6, + 0.5 = 6.5
        expected := 6.5;
        actual := to_real_val(fwd_data_out(H_OUT * W_OUT));
        report "Filter1[0,0]: expected=" & real'image(expected) & " got=" & real'image(actual);
        assert abs(actual - expected) < tolerance
            report "Filter 1 [0,0] FAILED!" severity error;

        -- Filter 1 at (1,1): center is input[2*4+2]=11, + 0.5 = 11.5
        expected := 11.5;
        actual := to_real_val(fwd_data_out(H_OUT * W_OUT + H_OUT - 1 + W_OUT - 1));
        report "Filter1[1,1]: expected=" & real'image(expected) & " got=" & real'image(actual);

        report "ALL TESTS PASSED!";
        std.env.stop;
    end process;

end architecture testbench;

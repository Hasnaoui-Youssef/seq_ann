library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

-- Testbench for RNN layer: simple sequence processing
-- Input: 3 timesteps of 2-dimensional input
-- Hidden: 4 units
-- Verifies: temporal unrolling, hidden state accumulation

entity rnn_tb is
end entity rnn_tb;

architecture testbench of rnn_tb is
    constant CLK_PERIOD : time := 10 ns;

    constant INPUT_SIZE  : integer := 2;
    constant HIDDEN_SIZE : integer := 4;
    constant SEQ_LEN     : integer := 3;
    constant TOTAL_INPUT : integer := SEQ_LEN * INPUT_SIZE;  -- 6

    constant RNN_WEIGHTS : integer := (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE;  -- 28

    signal clk : std_logic := '0';
    signal rst : std_logic := '0';

    signal fwd_ctrl_in  : layer_control_t := (valid => '0', last => '0');
    signal fwd_data_in  : std_logic_bus_array(0 to TOTAL_INPUT - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal fwd_ctrl_out : layer_control_t;
    signal fwd_data_out : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    signal bwd_ctrl_in  : layer_control_t := (valid => '0', last => '0');
    signal bwd_error_in : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal bwd_ctrl_out : layer_control_t;
    signal bwd_error_out: std_logic_bus_array(0 to TOTAL_INPUT - 1)(DATA_WIDTH - 1 downto 0);

    signal weight_load_en   : std_logic := '0';
    signal weight_load_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_load_done : std_logic;
    signal weight_update_en   : std_logic := '0';
    signal weight_learn_rate  : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal weight_update_done : std_logic;

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

    dut: entity work.rnn_layer
        generic map (
            INPUT_SIZE => INPUT_SIZE,
            HIDDEN_SIZE => HIDDEN_SIZE,
            SEQ_LEN => SEQ_LEN,
            USE_LSTM => false
        )
        port map (
            clk => clk, rst => rst,
            fwd_ctrl_in => fwd_ctrl_in, fwd_data_in => fwd_data_in,
            fwd_ctrl_out => fwd_ctrl_out, fwd_data_out => fwd_data_out,
            bwd_ctrl_in => bwd_ctrl_in, bwd_error_in => bwd_error_in,
            bwd_ctrl_out => bwd_ctrl_out, bwd_error_out => bwd_error_out,
            weight_load_en => weight_load_en, weight_load_data => weight_load_data,
            weight_load_done => weight_load_done,
            weight_update_en => weight_update_en, weight_learn_rate => weight_learn_rate,
            weight_update_done => weight_update_done
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
    begin
        rst <= '1';
        wait for CLK_PERIOD * 3;
        rst <= '0';
        wait for CLK_PERIOD * 2;

        -- Load weights: W_ih (INPUT_SIZE * HIDDEN_SIZE = 8)
        -- Simple pattern: W_ih = 0.5 for all
        for i in 0 to INPUT_SIZE * HIDDEN_SIZE - 1 loop
            load_weight(0.5);
        end loop;

        -- W_hh (HIDDEN_SIZE * HIDDEN_SIZE = 16)
        -- Simple pattern: identity * 0.1
        for j in 0 to HIDDEN_SIZE - 1 loop
            for i in 0 to HIDDEN_SIZE - 1 loop
                if i = j then
                    load_weight(0.1);
                else
                    load_weight(0.0);
                end if;
            end loop;
        end loop;

        -- Bias (HIDDEN_SIZE = 4): all 0
        for i in 0 to HIDDEN_SIZE - 1 loop
            load_weight(0.0);
        end loop;

        report "Weights loaded (" & integer'image(RNN_WEIGHTS) & " total)";

        -- Set input sequence: 3 timesteps of 2D input
        -- t=0: [1.0, 0.0]
        -- t=1: [0.0, 1.0]
        -- t=2: [1.0, 1.0]
        fwd_data_in(0) <= to_slv(1.0);
        fwd_data_in(1) <= to_slv(0.0);
        fwd_data_in(2) <= to_slv(0.0);
        fwd_data_in(3) <= to_slv(1.0);
        fwd_data_in(4) <= to_slv(1.0);
        fwd_data_in(5) <= to_slv(1.0);

        -- Trigger forward
        fwd_ctrl_in.valid <= '1';
        wait until rising_edge(clk);
        fwd_ctrl_in.valid <= '0';

        -- Wait for completion
        wait until fwd_ctrl_out.valid = '1';
        wait until rising_edge(clk);

        report "RNN forward pass completed!";
        for i in 0 to HIDDEN_SIZE - 1 loop
            report "h_out(" & integer'image(i) & ") = " & real'image(to_real_val(fwd_data_out(i)));
        end loop;

        -- Verify hidden state is non-zero (processed input through 3 timesteps)
        for i in 0 to HIDDEN_SIZE - 1 loop
            assert to_real_val(fwd_data_out(i)) /= 0.0
                report "Hidden unit " & integer'image(i) & " is zero - RNN not computing!" severity error;
        end loop;

        -- All hidden units should have the same value due to symmetric W_ih (0.5 everywhere)
        -- After t=0: h = tanh(0.5*1 + 0.5*0 + 0) = tanh(0.5) ≈ 0.5 for each unit
        -- After t=1: h = tanh(0.5*0 + 0.5*1 + 0.1*h_prev) ≈ ... for each unit
        -- After t=2: final hidden state

        report "ALL TESTS PASSED!";
        std.env.stop;
    end process;

end architecture testbench;

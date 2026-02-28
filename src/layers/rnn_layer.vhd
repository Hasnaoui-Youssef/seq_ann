library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

-- RNN Layer: Wraps an RNN cell with temporal unrolling
--
-- Accepts a sequence of SEQ_LEN timesteps, feeds them one-by-one
-- to the RNN cell, and outputs the final hidden state (or all states).
--
-- Input layout: [x_0_0, x_0_1, ..., x_0_{I-1}, x_1_0, ..., x_{T-1}_{I-1}]
-- Total input size: SEQ_LEN * INPUT_SIZE
-- Output: HIDDEN_SIZE (final hidden state)
--
-- Weight management: delegated to internal weight_bank

entity rnn_layer is
    generic (
        INPUT_SIZE  : integer := 4;
        HIDDEN_SIZE : integer := 8;
        SEQ_LEN     : integer := 10;
        USE_LSTM    : boolean := false
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Forward path
        fwd_ctrl_in  : in  layer_control_t;
        fwd_data_in  : in  std_logic_bus_array(0 to SEQ_LEN * INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0);
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);

        -- Backward path
        bwd_ctrl_in  : in  layer_control_t;
        bwd_error_in : in  std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out std_logic_bus_array(0 to SEQ_LEN * INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0);

        -- Weight management
        weight_load_en   : in  std_logic;
        weight_load_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_load_done : out std_logic;

        weight_update_en   : in  std_logic;
        weight_learn_rate  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_update_done : out std_logic
    );
end entity rnn_layer;

architecture rtl of rnn_layer is

    -- Weight count depends on cell type
    constant RNN_WEIGHTS  : integer := (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE;
    constant LSTM_WEIGHTS : integer := 4 * (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE;

    function sel_weights return integer is
    begin
        if USE_LSTM then return LSTM_WEIGHTS;
        else return RNN_WEIGHTS;
        end if;
    end function;

    constant NUM_WEIGHTS : integer := sel_weights;

    -- State machine for temporal unrolling
    type state_t is (IDLE, STEP, WAIT_STEP, DONE_STATE);
    signal state : state_t := IDLE;

    signal step_idx : integer range 0 to SEQ_LEN - 1 := 0;

    -- Hidden / cell state registers
    signal h_state : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal c_state : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

    -- Timestep input slice
    signal x_step : std_logic_bus_array(0 to INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

    -- Cell outputs
    signal h_new : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);
    signal c_new : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    -- Cell control
    signal cell_en   : std_logic := '0';
    signal cell_done : std_logic;

    -- Weight bank
    signal wb_weights : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);
    signal wb_read_en : std_logic := '0';

    -- Stored input for backward pass
    signal stored_input : std_logic_bus_array(0 to SEQ_LEN * INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

begin

    -- Weight bank instance
    wb_inst: entity work.weight_bank
        generic map (NUM_WEIGHTS => NUM_WEIGHTS)
        port map (
            clk => clk, rst => rst,
            load_en => weight_load_en,
            load_data => weight_load_data,
            load_done => weight_load_done,
            load_idx => open,
            weights_o => wb_weights,
            update_en => weight_update_en,
            grad_data => (0 to NUM_WEIGHTS - 1 => (others => '0')),
            learn_rate => weight_learn_rate,
            update_done => weight_update_done,
            save_en => '0',
            save_data => open,
            save_done => open,
            save_idx => open
        );

    -- RNN cell (non-LSTM)
    gen_rnn: if not USE_LSTM generate
        rnn_inst: entity work.rnn_cell
            generic map (INPUT_SIZE => INPUT_SIZE, HIDDEN_SIZE => HIDDEN_SIZE)
            port map (
                clk => clk, rst => rst,
                x_in => x_step, h_prev => h_state,
                h_out => h_new,
                weights => wb_weights(0 to RNN_WEIGHTS - 1),
                compute_en => cell_en, compute_done => cell_done
            );
    end generate;

    -- LSTM cell
    gen_lstm: if USE_LSTM generate
        lstm_inst: entity work.lstm_cell
            generic map (INPUT_SIZE => INPUT_SIZE, HIDDEN_SIZE => HIDDEN_SIZE)
            port map (
                clk => clk, rst => rst,
                x_in => x_step, h_prev => h_state, c_prev => c_state,
                h_out => h_new, c_out => c_new,
                weights => wb_weights(0 to LSTM_WEIGHTS - 1),
                compute_en => cell_en, compute_done => cell_done
            );
    end generate;

    -- Extract timestep slice
    process(fwd_data_in, step_idx)
    begin
        for i in 0 to INPUT_SIZE - 1 loop
            x_step(i) <= fwd_data_in(step_idx * INPUT_SIZE + i);
        end loop;
    end process;

    -- Output final hidden state
    fwd_data_out <= h_state;

    -- Backward pass: simple passthrough (BPTT not implemented yet)
    bwd_error_out <= (others => (others => '0'));

    -- Temporal unrolling FSM
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state <= IDLE;
                step_idx <= 0;
                h_state <= (others => (others => '0'));
                c_state <= (others => (others => '0'));
                cell_en <= '0';
                wb_read_en <= '0';
                fwd_ctrl_out <= (valid => '0', last => '0');
                bwd_ctrl_out <= (valid => '0', last => '0');
            else
                cell_en <= '0';
                fwd_ctrl_out.valid <= '0';

                case state is
                    when IDLE =>
                        if fwd_ctrl_in.valid = '1' then
                            step_idx <= 0;
                            h_state <= (others => (others => '0'));
                            c_state <= (others => (others => '0'));
                            wb_read_en <= '1';
                            stored_input <= fwd_data_in;
                            state <= STEP;
                        end if;

                        if bwd_ctrl_in.valid = '1' then
                            bwd_ctrl_out.valid <= '1';
                            bwd_ctrl_out.last <= bwd_ctrl_in.last;
                        end if;

                    when STEP =>
                        cell_en <= '1';
                        state <= WAIT_STEP;

                    when WAIT_STEP =>
                        if cell_done = '1' then
                            h_state <= h_new;
                            if USE_LSTM then
                                c_state <= c_new;
                            end if;

                            if step_idx = SEQ_LEN - 1 then
                                state <= DONE_STATE;
                            else
                                step_idx <= step_idx + 1;
                                state <= STEP;
                            end if;
                        end if;

                    when DONE_STATE =>
                        fwd_ctrl_out.valid <= '1';
                        fwd_ctrl_out.last <= '1';
                        wb_read_en <= '0';
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end architecture rtl;

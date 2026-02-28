library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

-- RNN Cell: Single recurrent computation step
--
-- h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
--
-- Weight layout in weight_bank:
--   [0 .. INPUT_SIZE*HIDDEN_SIZE-1]                           : W_ih (input-to-hidden)
--   [INPUT_SIZE*HIDDEN_SIZE .. (INPUT_SIZE+HIDDEN_SIZE)*HIDDEN_SIZE-1] : W_hh (hidden-to-hidden)
--   [(INPUT_SIZE+HIDDEN_SIZE)*HIDDEN_SIZE .. NUM_WEIGHTS-1]    : bias

entity rnn_cell is
    generic (
        INPUT_SIZE  : integer := 4;
        HIDDEN_SIZE : integer := 8
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Input for this timestep
        x_in  : in std_logic_bus_array(0 to INPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0);
        -- Previous hidden state
        h_prev : in std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);
        -- New hidden state
        h_out  : out std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);

        -- Weights (from weight bank, all available in parallel)
        weights : in std_logic_bus_array(0 to (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0);

        -- Control
        compute_en : in std_logic;
        compute_done : out std_logic
    );
end entity rnn_cell;

architecture rtl of rnn_cell is

    constant W_IH_SIZE : integer := INPUT_SIZE * HIDDEN_SIZE;
    constant W_HH_SIZE : integer := HIDDEN_SIZE * HIDDEN_SIZE;
    constant NUM_WEIGHTS : integer := (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE;

    function mult(a, b : std_logic_vector) return sfixed_bus is
    begin
        return resize(to_sfixed(a, INT_BITS - 1, -FRAC_BITS) *
                      to_sfixed(b, INT_BITS - 1, -FRAC_BITS),
                      INT_BITS - 1, -FRAC_BITS);
    end function;

    -- Tanh approximation using piecewise linear
    function tanh_approx(x : sfixed_bus) return sfixed_bus is
        variable one_pos : sfixed_bus := to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS);
        variable one_neg : sfixed_bus := to_sfixed(-1.0, INT_BITS - 1, -FRAC_BITS);
    begin
        if x > one_pos then
            return one_pos;
        elsif x < one_neg then
            return one_neg;
        else
            return x;  -- Linear region: tanh(x) ≈ x for small x
        end if;
    end function;

    signal h_reg : std_logic_bus_array(0 to HIDDEN_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal done_reg : std_logic := '0';

begin

    h_out <= h_reg;
    compute_done <= done_reg;

    process(clk)
        variable acc : sfixed_bus;
        variable w_idx : integer;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                h_reg <= (others => (others => '0'));
                done_reg <= '0';
            else
                done_reg <= '0';
                if compute_en = '1' then
                    for j in 0 to HIDDEN_SIZE - 1 loop
                        -- Start with bias
                        w_idx := W_IH_SIZE + W_HH_SIZE + j;
                        acc := to_sfixed(weights(w_idx), INT_BITS - 1, -FRAC_BITS);

                        -- W_ih * x_t
                        for i in 0 to INPUT_SIZE - 1 loop
                            w_idx := j * INPUT_SIZE + i;
                            acc := resize(acc + mult(x_in(i), weights(w_idx)),
                                         INT_BITS - 1, -FRAC_BITS);
                        end loop;

                        -- W_hh * h_{t-1}
                        for i in 0 to HIDDEN_SIZE - 1 loop
                            w_idx := W_IH_SIZE + j * HIDDEN_SIZE + i;
                            acc := resize(acc + mult(h_prev(i), weights(w_idx)),
                                         INT_BITS - 1, -FRAC_BITS);
                        end loop;

                        -- Apply tanh activation
                        h_reg(j) <= to_std_logic_vector(tanh_approx(acc));
                    end loop;
                    done_reg <= '1';
                end if;
            end if;
        end if;
    end process;

end architecture rtl;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

-- LSTM Cell: Long Short-Term Memory computation step
--
-- Gates:
--   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   forget gate
--   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   input gate
--   g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)      candidate
--   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)    output gate
--
--   c_t = f_t * c_{t-1} + i_t * g_t
--   h_t = o_t * tanh(c_t)
--
-- Weight layout: 4 sets of [(INPUT_SIZE + HIDDEN_SIZE) weights + 1 bias] per hidden unit
-- Total: 4 * (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE

entity lstm_cell is
    generic (
        INPUT_SIZE  : integer := 4;
        HIDDEN_SIZE : integer := 8
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        x_in   : in sfixed_bus_array(0 to INPUT_SIZE - 1);
        h_prev : in sfixed_bus_array(0 to HIDDEN_SIZE - 1);
        c_prev : in sfixed_bus_array(0 to HIDDEN_SIZE - 1);

        h_out  : out sfixed_bus_array(0 to HIDDEN_SIZE - 1);
        c_out  : out sfixed_bus_array(0 to HIDDEN_SIZE - 1);

        weights : in sfixed_bus_array(0 to 4 * (INPUT_SIZE + HIDDEN_SIZE + 1) * HIDDEN_SIZE - 1);

        compute_en   : in std_logic;
        compute_done : out std_logic
    );
end entity lstm_cell;

architecture rtl of lstm_cell is

    constant CONCAT_SIZE : integer := INPUT_SIZE + HIDDEN_SIZE;
    constant GATE_WEIGHTS : integer := (CONCAT_SIZE + 1) * HIDDEN_SIZE;

    function mult_sf(a, b : sfixed_bus) return sfixed_bus is
    begin
        return resize(a * b, INT_BITS - 1, -FRAC_BITS);
    end function;

    -- Sigmoid approximation: piecewise linear
    function sigmoid_approx(x : sfixed_bus) return sfixed_bus is
        constant half    : sfixed_bus := to_sfixed(0.5, INT_BITS - 1, -FRAC_BITS);
        constant one     : sfixed_bus := to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS);
        constant zero    : sfixed_bus := to_sfixed(0.0, INT_BITS - 1, -FRAC_BITS);
        constant quarter : sfixed_bus := to_sfixed(0.25, INT_BITS - 1, -FRAC_BITS);
        constant two_pos : sfixed_bus := to_sfixed(2.0, INT_BITS - 1, -FRAC_BITS);
        constant two_neg : sfixed_bus := to_sfixed(-2.0, INT_BITS - 1, -FRAC_BITS);
    begin
        if x > two_pos then
            return one;
        elsif x < two_neg then
            return zero;
        else
            return resize(half + mult_sf(quarter, x), INT_BITS - 1, -FRAC_BITS);
        end if;
    end function;

    -- Tanh approximation: piecewise linear
    function tanh_approx(x : sfixed_bus) return sfixed_bus is
        constant one_pos : sfixed_bus := to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS);
        constant one_neg : sfixed_bus := to_sfixed(-1.0, INT_BITS - 1, -FRAC_BITS);
    begin
        if x > one_pos then
            return one_pos;
        elsif x < one_neg then
            return one_neg;
        else
            return x;
        end if;
    end function;

    signal h_reg : sfixed_bus_array(0 to HIDDEN_SIZE - 1)
        := (others => (others => '0'));
    signal c_reg : sfixed_bus_array(0 to HIDDEN_SIZE - 1)
        := (others => (others => '0'));
    signal done_reg : std_logic := '0';

    -- Compute gate activation for one hidden unit
    -- gate_offset: 0=forget, 1=input, 2=candidate, 3=output
    function compute_gate(
        gate_offset : integer;
        hidden_idx  : integer;
        x           : sfixed_bus_array;
        h           : sfixed_bus_array;
        w           : sfixed_bus_array
    ) return sfixed_bus is
        variable acc : sfixed_bus;
        variable w_base : integer;
        variable w_idx : integer;
    begin
        w_base := gate_offset * GATE_WEIGHTS + hidden_idx * (CONCAT_SIZE + 1);
        -- Bias
        acc := w(w_base + CONCAT_SIZE);
        -- Input weights
        for i in 0 to INPUT_SIZE - 1 loop
            acc := resize(acc + mult_sf(x(i), w(w_base + i)),
                         INT_BITS - 1, -FRAC_BITS);
        end loop;
        -- Hidden weights
        for i in 0 to HIDDEN_SIZE - 1 loop
            acc := resize(acc + mult_sf(h(i), w(w_base + INPUT_SIZE + i)),
                         INT_BITS - 1, -FRAC_BITS);
        end loop;
        return acc;
    end function;

begin

    h_out <= h_reg;
    c_out <= c_reg;
    compute_done <= done_reg;

    process(clk)
        variable f_gate, i_gate, g_gate, o_gate : sfixed_bus;
        variable c_new, c_tanh, h_new : sfixed_bus;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                h_reg <= (others => (others => '0'));
                c_reg <= (others => (others => '0'));
                done_reg <= '0';
            else
                done_reg <= '0';
                if compute_en = '1' then
                    for j in 0 to HIDDEN_SIZE - 1 loop
                        -- Compute gates
                        f_gate := sigmoid_approx(compute_gate(0, j, x_in, h_prev, weights));
                        i_gate := sigmoid_approx(compute_gate(1, j, x_in, h_prev, weights));
                        g_gate := tanh_approx(compute_gate(2, j, x_in, h_prev, weights));
                        o_gate := sigmoid_approx(compute_gate(3, j, x_in, h_prev, weights));

                        -- Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
                        c_new := resize(
                            mult_sf(f_gate, c_prev(j)) +
                            mult_sf(i_gate, g_gate),
                            INT_BITS - 1, -FRAC_BITS);
                        c_reg(j) <= c_new;

                        -- Hidden state: h_t = o_t * tanh(c_t)
                        c_tanh := tanh_approx(c_new);
                        h_new := mult_sf(o_gate, c_tanh);
                        h_reg(j) <= h_new;
                    end loop;
                    done_reg <= '1';
                end if;
            end if;
        end if;
    end process;

end architecture rtl;

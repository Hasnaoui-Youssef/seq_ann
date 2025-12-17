library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use IEEE.fixed_float_types.all;
use work.types.all;
use work.sigmoid_lut_pkg.all;

entity neuron is
    generic(
        NUM_INPUTS : integer := 4;
        USE_SIGMOID : boolean := true
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Forward Pass
        fwd_en       : in std_logic;
        inputs_i     : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        weights_i    : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        bias_i       : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_o     : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Backward Pass
        bwd_en       : in std_logic;
        error_i      : in std_logic_vector(DATA_WIDTH - 1 downto 0); -- dL/dy

        -- Gradients Output
        grad_weights_o : out std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        grad_bias_o    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        grad_inputs_o  : out std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0) -- dL/dx
    );
end entity neuron;

architecture rtl of neuron is

    -- Accumulator configuration: NUM_INPUTS products + 1 bias = NUM_INPUTS + 1 terms
    constant ACC_SIZE : integer := NUM_INPUTS + 1;
    -- Accumulator output width: ACC_SIZE + DATA_WIDTH
    constant ACC_OUT_WIDTH : integer := ACC_SIZE + DATA_WIDTH;

    -- Internal Storage for Forward Pass (needed for Backward)
    signal stored_inputs : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
    signal stored_output_reg : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal stored_deriv  : std_logic_vector(DATA_WIDTH - 1 downto 0); -- f'(net)

    -- Multiplication products (input * weight for each input)
    signal products : std_logic_bus_array(0 to NUM_INPUTS - 1)(2 * DATA_WIDTH - 1 downto 0);

    -- Accumulator inputs: products resized + bias
    signal acc_inputs : std_logic_bus_array(ACC_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);

    -- Accumulator output
    signal acc_sum : std_logic_vector(ACC_OUT_WIDTH - 1 downto 0);
    -- Note: acc_overflow is intentionally not used in this design.
    -- The accumulator is sized (ACC_SIZE + DATA_WIDTH bits) to prevent overflow
    -- for the expected range of inputs, so overflow handling is not required.
    signal acc_overflow : std_logic;

    -- Activation function output
    signal act_out_sig : sfixed_bus;

    -- Helper for fixed-point multiplication (for backward pass)
    function mult(a, b : std_logic_vector) return std_logic_vector is
        variable res : sfixed_bus;
    begin
        res := resize(to_sfixed(a, INT_BITS - 1, -FRAC_BITS) *
                      to_sfixed(b, INT_BITS - 1, -FRAC_BITS),
                      INT_BITS - 1, -FRAC_BITS);
        return to_std_logic_vector(res);
    end function;

    -- Helper for sigmoid derivative: y * (1 - y)
    function calc_sigmoid_deriv(y : std_logic_vector) return std_logic_vector is
        variable y_sf : sfixed_bus;
        variable one_sf : sfixed_bus;
        variable res : sfixed_bus;
    begin
        y_sf := to_sfixed(y, INT_BITS - 1, -FRAC_BITS);
        one_sf := to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS);
        res := resize(y_sf * (one_sf - y_sf), INT_BITS - 1, -FRAC_BITS);
        return to_std_logic_vector(res);
    end function;

begin

    -- Output is registered value
    output_o <= stored_output_reg;

    -- Generate multipliers for each input*weight pair
    gen_mult: for i in 0 to NUM_INPUTS - 1 generate
        u_mult: entity work.mult
            port map (
                a_i => inputs_i(i),
                b_i => weights_i(i),
                product_o => products(i)
            );
    end generate;

    -- Prepare accumulator inputs:
    -- Products need to be resized from 2*DATA_WIDTH to DATA_WIDTH (take middle bits for fixed-point)
    -- For Q(INT_BITS).(FRAC_BITS) * Q(INT_BITS).(FRAC_BITS) = Q(2*INT_BITS).(2*FRAC_BITS)
    -- We want Q(INT_BITS).(FRAC_BITS), so take bits [2*DATA_WIDTH - INT_BITS - 1 downto FRAC_BITS]
    gen_acc_inputs: for i in 0 to NUM_INPUTS - 1 generate
        acc_inputs(i) <= products(i)(DATA_WIDTH + FRAC_BITS - 1 downto FRAC_BITS);
    end generate;
    -- Last accumulator input is bias
    acc_inputs(NUM_INPUTS) <= bias_i;

    -- Instantiate accumulator
    u_acc: entity work.acc
        generic map (
            size => ACC_SIZE,
            signed_acc => true
        )
        port map (
            inputs => acc_inputs,
            sum_o => acc_sum,
            overflow_o => acc_overflow
        );

    -- Instantiate Activation Function
    -- Input width is ACC_OUT_WIDTH, output is DATA_WIDTH (via types package)
    u_act: entity work.activation_func(sigmoid)
        generic map (
            input_width => ACC_OUT_WIDTH
        )
        port map (
            input_i => acc_sum,
            output_o => act_out_sig
        );

    process(clk, rst)
        variable delta : std_logic_vector(DATA_WIDTH - 1 downto 0); -- dL/dy * f'(net)
    begin
        if rst = '1' then
            stored_output_reg <= (others => '0');
            stored_deriv <= (others => '0');
            grad_bias_o <= (others => '0');
            grad_weights_o <= (others => (others => '0'));
            grad_inputs_o <= (others => (others => '0'));
            stored_inputs <= (others => (others => '0'));

        elsif rising_edge(clk) then

            -- Forward Pass
            if fwd_en = '1' then
                -- Store inputs for backward pass
                stored_inputs <= inputs_i;

                -- Latch Output
                if USE_SIGMOID then
                    stored_output_reg <= to_std_logic_vector(act_out_sig);
                else
                    -- For non-sigmoid, resize accumulator output to DATA_WIDTH
                    -- Use saturating resize to handle potential overflow gracefully
                    stored_output_reg <= to_std_logic_vector(
                        resize(
                            to_sfixed(acc_sum, ACC_OUT_WIDTH-1, -FRAC_BITS),
                            INT_BITS - 1,
                            -FRAC_BITS,
                            fixed_saturate,
                            fixed_truncate
                        )
                    );
                end if;

                -- Derivative: y * (1 - y) for sigmoid, 1.0 for linear
                if USE_SIGMOID then
                    -- Use the current activation output (act_out_sig) for derivative calculation
                    stored_deriv <= calc_sigmoid_deriv(to_std_logic_vector(act_out_sig));
                else
                    -- For non-sigmoid (linear), derivative is 1.0
                    stored_deriv <= to_std_logic_vector(to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS));
                end if;
            end if;

            -- Backward Pass
            if bwd_en = '1' then
                -- Calculate Delta = Error * Derivative
                delta := mult(error_i, stored_deriv);

                -- Gradient for Bias: dL/db = delta
                grad_bias_o <= delta;

                -- Gradients for Weights: dL/dw = delta * input
                -- Gradients for Inputs: dL/dx = delta * weight
                for i in 0 to NUM_INPUTS - 1 loop
                    grad_weights_o(i) <= mult(delta, stored_inputs(i));
                    grad_inputs_o(i) <= mult(delta, weights_i(i));
                end loop;
            end if;

        end if;
    end process;

end architecture rtl;

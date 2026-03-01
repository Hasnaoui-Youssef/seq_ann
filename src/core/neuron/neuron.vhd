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
        inputs_i     : in sfixed_bus_array(0 to NUM_INPUTS - 1);
        weights_i    : in sfixed_bus_array(0 to NUM_INPUTS - 1);
        bias_i       : in sfixed_bus;
        output_o     : out sfixed_bus;

        -- Backward Pass
        bwd_en       : in std_logic;
        error_i      : in sfixed_bus; -- dL/dy

        -- Gradients Output
        grad_weights_o : out sfixed_bus_array(0 to NUM_INPUTS - 1);
        grad_bias_o    : out sfixed_bus;
        grad_inputs_o  : out sfixed_bus_array(0 to NUM_INPUTS - 1) -- dL/dx
    );
end entity neuron;

architecture rtl of neuron is

    -- Internal Storage for Forward Pass (needed for Backward)
    signal stored_inputs : sfixed_bus_array(0 to NUM_INPUTS - 1)
        := (others => (others => '0'));
    signal stored_output_reg : sfixed_bus := (others => '0');
    signal stored_deriv  : sfixed_bus; -- f'(net)

    -- Multiplication products
    signal products : sfixed_bus_array(0 to NUM_INPUTS - 1);

    -- Accumulator: sum of products + bias
    signal acc_sum : sfixed_bus;

    -- Activation function output
    signal act_out_sig : sfixed_bus;

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

    -- Combinational accumulation: sum all products + bias
    process(all)
        variable sum : sfixed_bus;
    begin
        sum := bias_i;
        for i in 0 to NUM_INPUTS - 1 loop
            sum := resize(sum + products(i), INT_BITS - 1, -FRAC_BITS);
        end loop;
        acc_sum <= sum;
    end process;

    -- Instantiate Activation Function
    u_act: entity work.activation_func(sigmoid)
        generic map (
            input_int_bits => INT_BITS
        )
        port map (
            input_i => acc_sum,
            output_o => act_out_sig
        );

    process(clk, rst)
        variable delta : sfixed_bus;
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
                stored_inputs <= inputs_i;

                if USE_SIGMOID then
                    stored_output_reg <= act_out_sig;
                    -- Derivative: y * (1 - y) for sigmoid
                    stored_deriv <= resize(
                        act_out_sig * (to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS) - act_out_sig),
                        INT_BITS - 1, -FRAC_BITS);
                else
                    -- Linear: just resize accumulator sum
                    stored_output_reg <= acc_sum;
                    stored_deriv <= to_sfixed(1.0, INT_BITS - 1, -FRAC_BITS);
                end if;
            end if;

            -- Backward Pass
            if bwd_en = '1' then
                delta := resize(error_i * stored_deriv, INT_BITS - 1, -FRAC_BITS);
                grad_bias_o <= delta;
                for i in 0 to NUM_INPUTS - 1 loop
                    grad_weights_o(i) <= resize(delta * stored_inputs(i), INT_BITS - 1, -FRAC_BITS);
                    grad_inputs_o(i) <= resize(delta * weights_i(i), INT_BITS - 1, -FRAC_BITS);
                end loop;
            end if;

        end if;
    end process;

end architecture rtl;

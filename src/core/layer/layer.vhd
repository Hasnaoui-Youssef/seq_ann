library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;
use IEEE.fixed_pkg.all;

entity layer is
    generic(
        NUM_INPUTS : integer := 4;
        LAYER_SIZE : integer := 3;
        USE_SIGMOID : boolean := true
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Control
        fwd_en : in std_logic;
        bwd_en : in std_logic;

        -- Forward Interface
        fwd_ctrl_in  : in layer_control_t;
        fwd_data_in  : in sfixed_bus_array(0 to NUM_INPUTS - 1);

        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out sfixed_bus_array(0 to LAYER_SIZE - 1);

        -- Backward Interface
        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in sfixed_bus_array(0 to LAYER_SIZE - 1);

        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out sfixed_bus_array(0 to NUM_INPUTS - 1);

        -- Weight Bank Interface
        weight_load_en   : in  std_logic;
        weight_load_data : in  sfixed_bus;
        weight_load_done : out std_logic;

        weight_save_en   : in  std_logic;
        weight_save_data : out sfixed_bus;
        weight_save_done : out std_logic;

        -- Gradient update (training)
        weight_update_en   : in  std_logic;
        weight_learn_rate  : in  sfixed_bus;
        weight_update_done : out std_logic;

        -- Gradients Output
        grads_out : out sfixed_bus_array(0 to (NUM_INPUTS + 1) * LAYER_SIZE - 1)
    );
end entity layer;

architecture rtl of layer is

    constant NUM_WEIGHTS : integer := (NUM_INPUTS + 1) * LAYER_SIZE;

    signal neuron_outputs : sfixed_bus_array(0 to LAYER_SIZE - 1);

    signal weights_internal : sfixed_bus_array(0 to NUM_WEIGHTS - 1);

    signal grads_internal : sfixed_bus_array(0 to NUM_WEIGHTS - 1);

    type input_grad_array is array (0 to LAYER_SIZE - 1) of sfixed_bus_array(0 to NUM_INPUTS - 1);
    signal input_grads : input_grad_array;

    function sum_input_grads(input_idx : integer; grads : input_grad_array) return sfixed_bus is
        variable sum : sfixed_bus := (others => '0');
    begin
        for i in 0 to LAYER_SIZE - 1 loop
            sum := resize(sum + grads(i)(input_idx), sum);
        end loop;
        return sum;
    end function;

begin

    u_weight_bank: entity work.weight_bank
        generic map (
            NUM_WEIGHTS => NUM_WEIGHTS
        )
        port map (
            clk => clk,
            rst => rst,
            load_en => weight_load_en,
            load_data => weight_load_data,
            load_done => weight_load_done,
            load_idx => open,
            weights_o => weights_internal,
            update_en => weight_update_en,
            grad_data => grads_internal,
            learn_rate => weight_learn_rate,
            update_done => weight_update_done,
            save_en => weight_save_en,
            save_data => weight_save_data,
            save_done => weight_save_done,
            save_idx => open
        );

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                fwd_ctrl_out.valid <= '0';
                fwd_ctrl_out.last <= '0';
                bwd_ctrl_out.valid <= '0';
                bwd_ctrl_out.last <= '0';
            else
                fwd_ctrl_out <= fwd_ctrl_in;
                bwd_ctrl_out <= bwd_ctrl_in;
            end if;
        end if;
    end process;

    fwd_data_out <= neuron_outputs;
    grads_out <= grads_internal;

    gen_neurons: for i in 0 to LAYER_SIZE - 1 generate
        constant w_start : integer := i * (NUM_INPUTS + 1);

        signal n_weights : sfixed_bus_array(0 to NUM_INPUTS - 1);
        signal n_bias    : sfixed_bus;

        signal n_grad_weights : sfixed_bus_array(0 to NUM_INPUTS - 1);
        signal n_grad_bias    : sfixed_bus;

    begin
        assign_w: for j in 0 to NUM_INPUTS - 1 generate
            n_weights(j) <= weights_internal(w_start + j);
        end generate;
        n_bias <= weights_internal(w_start + NUM_INPUTS);

        assign_g: for j in 0 to NUM_INPUTS - 1 generate
            grads_internal(w_start + j) <= n_grad_weights(j);
        end generate;
        grads_internal(w_start + NUM_INPUTS) <= n_grad_bias;

        u_neuron: entity work.neuron
            generic map (
                NUM_INPUTS => NUM_INPUTS,
                USE_SIGMOID => USE_SIGMOID
            )
            port map (
                clk => clk,
                rst => rst,
                fwd_en => fwd_en,
                inputs_i => fwd_data_in,
                weights_i => n_weights,
                bias_i => n_bias,
                output_o => neuron_outputs(i),
                bwd_en => bwd_en,
                error_i => bwd_error_in(i),
                grad_weights_o => n_grad_weights,
                grad_bias_o => n_grad_bias,
                grad_inputs_o => input_grads(i)
            );
    end generate;

    -- Backward Error Aggregation (dL/dx to previous layer)
    gen_bwd_out: for j in 0 to NUM_INPUTS - 1 generate
        bwd_error_out(j) <= sum_input_grads(j, input_grads);
    end generate;

end architecture rtl;

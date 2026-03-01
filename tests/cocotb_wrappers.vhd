-- Thin wrappers for cocotb testbenches.
-- These bind specific architectures, set generics, and provide
-- std_logic_vector ports that cocotb/VPI can access reliably.
-- Internal conversions to/from sfixed happen inside the wrapper.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.fixed_pkg.all;
use work.types.all;

-- Activation function wrapper (binds sigmoid architecture)
entity activation_func_wrap is
    port (
        input_i  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_o : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity activation_func_wrap;

architecture wrap of activation_func_wrap is
    signal in_sf  : sfixed(INT_BITS - 1 downto -FRAC_BITS);
    signal out_sf : sfixed_bus;
begin
    in_sf <= to_sfixed(input_i, INT_BITS - 1, -FRAC_BITS);
    u_act : entity work.activation_func(sigmoid)
        generic map (input_int_bits => INT_BITS)
        port map (input_i => in_sf, output_o => out_sf);
    output_o <= to_std_logic_vector(out_sf);
end architecture wrap;

-- Neuron wrapper (4 inputs, sigmoid)
-- Array ports are flattened to concatenated std_logic_vectors.
-- inputs_flat(0 to 4*DATA_WIDTH-1): inputs_i(0) is MSB portion, inputs_i(3) is LSB portion.
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity neuron_wrap is
    port (
        clk : in std_logic;
        rst : in std_logic;
        fwd_en       : in std_logic;
        inputs_flat  : in  std_logic_vector(4 * DATA_WIDTH - 1 downto 0);
        weights_flat : in  std_logic_vector(4 * DATA_WIDTH - 1 downto 0);
        bias_slv     : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_slv   : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        bwd_en       : in std_logic;
        error_slv    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        grad_weights_flat : out std_logic_vector(4 * DATA_WIDTH - 1 downto 0);
        grad_bias_slv     : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        grad_inputs_flat  : out std_logic_vector(4 * DATA_WIDTH - 1 downto 0)
    );
end entity neuron_wrap;

architecture wrap of neuron_wrap is
    constant N : integer := 4;
    signal inputs_sf  : sfixed_bus_array(0 to N - 1);
    signal weights_sf : sfixed_bus_array(0 to N - 1);
    signal bias_sf    : sfixed_bus;
    signal output_sf  : sfixed_bus;
    signal error_sf   : sfixed_bus;
    signal grad_w_sf  : sfixed_bus_array(0 to N - 1);
    signal grad_b_sf  : sfixed_bus;
    signal grad_i_sf  : sfixed_bus_array(0 to N - 1);
begin
    -- Unpack flat vectors to sfixed arrays
    gen_in: for i in 0 to N - 1 generate
        inputs_sf(i)  <= to_sfixed(inputs_flat((N - 1 - i + 1) * DATA_WIDTH - 1 downto (N - 1 - i) * DATA_WIDTH), INT_BITS - 1, -FRAC_BITS);
        weights_sf(i) <= to_sfixed(weights_flat((N - 1 - i + 1) * DATA_WIDTH - 1 downto (N - 1 - i) * DATA_WIDTH), INT_BITS - 1, -FRAC_BITS);
    end generate;
    bias_sf  <= to_sfixed(bias_slv, INT_BITS - 1, -FRAC_BITS);
    error_sf <= to_sfixed(error_slv, INT_BITS - 1, -FRAC_BITS);

    u_neuron : entity work.neuron
        generic map (num_inputs => N, use_sigmoid => true)
        port map (
            clk => clk, rst => rst,
            fwd_en => fwd_en,
            inputs_i => inputs_sf, weights_i => weights_sf, bias_i => bias_sf,
            output_o => output_sf,
            bwd_en => bwd_en, error_i => error_sf,
            grad_weights_o => grad_w_sf,
            grad_bias_o => grad_b_sf,
            grad_inputs_o => grad_i_sf
        );

    -- Pack outputs
    output_slv   <= to_std_logic_vector(output_sf);
    grad_bias_slv <= to_std_logic_vector(grad_b_sf);
    gen_out: for i in 0 to N - 1 generate
        grad_weights_flat((N - 1 - i + 1) * DATA_WIDTH - 1 downto (N - 1 - i) * DATA_WIDTH) <= to_std_logic_vector(grad_w_sf(i));
        grad_inputs_flat((N - 1 - i + 1) * DATA_WIDTH - 1 downto (N - 1 - i) * DATA_WIDTH)  <= to_std_logic_vector(grad_i_sf(i));
    end generate;
end architecture wrap;

-- Layer wrapper (4 inputs -> 2 neurons, sigmoid)
-- Array ports are flattened to concatenated std_logic_vectors.
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

entity layer_wrap is
    port (
        clk : in std_logic;
        rst : in std_logic;
        fwd_en : in std_logic;
        bwd_en : in std_logic;
        fwd_ctrl_in_valid  : in std_logic;
        fwd_ctrl_in_last   : in std_logic;
        fwd_data_in_flat   : in  std_logic_vector(4 * DATA_WIDTH - 1 downto 0);
        fwd_ctrl_out_valid : out std_logic;
        fwd_ctrl_out_last  : out std_logic;
        fwd_data_out_flat  : out std_logic_vector(2 * DATA_WIDTH - 1 downto 0);
        bwd_ctrl_in_valid  : in std_logic;
        bwd_ctrl_in_last   : in std_logic;
        bwd_error_in_flat  : in  std_logic_vector(2 * DATA_WIDTH - 1 downto 0);
        bwd_ctrl_out_valid : out std_logic;
        bwd_ctrl_out_last  : out std_logic;
        bwd_error_out_flat : out std_logic_vector(4 * DATA_WIDTH - 1 downto 0);
        weight_load_en   : in  std_logic;
        weight_load_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_load_done : out std_logic;
        weight_save_en   : in  std_logic;
        weight_save_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_save_done : out std_logic;
        weight_update_en   : in  std_logic;
        weight_learn_rate  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_update_done : out std_logic;
        grads_out_flat : out std_logic_vector(10 * DATA_WIDTH - 1 downto 0)
    );
end entity layer_wrap;

architecture wrap of layer_wrap is
    constant NI : integer := 4;
    constant NO : integer := 2;
    constant NW : integer := 10;  -- (4+1)*2

    signal fwd_ctrl_in  : layer_control_t;
    signal fwd_ctrl_out : layer_control_t;
    signal bwd_ctrl_in  : layer_control_t;
    signal bwd_ctrl_out : layer_control_t;

    signal fwd_in_sf   : sfixed_bus_array(0 to NI - 1);
    signal fwd_out_sf  : sfixed_bus_array(0 to NO - 1);
    signal bwd_in_sf   : sfixed_bus_array(0 to NO - 1);
    signal bwd_out_sf  : sfixed_bus_array(0 to NI - 1);
    signal wl_data_sf  : sfixed_bus;
    signal ws_data_sf  : sfixed_bus;
    signal lr_sf       : sfixed_bus;
    signal grads_sf    : sfixed_bus_array(0 to NW - 1);
begin
    -- Control record mapping
    fwd_ctrl_in.valid <= fwd_ctrl_in_valid;
    fwd_ctrl_in.last  <= fwd_ctrl_in_last;
    fwd_ctrl_out_valid <= fwd_ctrl_out.valid;
    fwd_ctrl_out_last  <= fwd_ctrl_out.last;
    bwd_ctrl_in.valid <= bwd_ctrl_in_valid;
    bwd_ctrl_in.last  <= bwd_ctrl_in_last;
    bwd_ctrl_out_valid <= bwd_ctrl_out.valid;
    bwd_ctrl_out_last  <= bwd_ctrl_out.last;

    -- Unpack inputs
    gen_fi: for i in 0 to NI - 1 generate
        fwd_in_sf(i) <= to_sfixed(fwd_data_in_flat((NI - 1 - i + 1) * DATA_WIDTH - 1 downto (NI - 1 - i) * DATA_WIDTH), INT_BITS - 1, -FRAC_BITS);
    end generate;
    gen_bi: for i in 0 to NO - 1 generate
        bwd_in_sf(i) <= to_sfixed(bwd_error_in_flat((NO - 1 - i + 1) * DATA_WIDTH - 1 downto (NO - 1 - i) * DATA_WIDTH), INT_BITS - 1, -FRAC_BITS);
    end generate;
    wl_data_sf <= to_sfixed(weight_load_data, INT_BITS - 1, -FRAC_BITS);
    lr_sf      <= to_sfixed(weight_learn_rate, INT_BITS - 1, -FRAC_BITS);

    u_layer : entity work.layer
        generic map (NUM_INPUTS => NI, LAYER_SIZE => NO, USE_SIGMOID => true)
        port map (
            clk => clk, rst => rst,
            fwd_en => fwd_en, bwd_en => bwd_en,
            fwd_ctrl_in => fwd_ctrl_in, fwd_data_in => fwd_in_sf,
            fwd_ctrl_out => fwd_ctrl_out, fwd_data_out => fwd_out_sf,
            bwd_ctrl_in => bwd_ctrl_in, bwd_error_in => bwd_in_sf,
            bwd_ctrl_out => bwd_ctrl_out, bwd_error_out => bwd_out_sf,
            weight_load_en => weight_load_en, weight_load_data => wl_data_sf,
            weight_load_done => weight_load_done,
            weight_save_en => weight_save_en, weight_save_data => ws_data_sf,
            weight_save_done => weight_save_done,
            weight_update_en => weight_update_en,
            weight_learn_rate => lr_sf,
            weight_update_done => weight_update_done,
            grads_out => grads_sf
        );

    -- Pack outputs
    weight_save_data <= to_std_logic_vector(ws_data_sf);
    gen_fo: for i in 0 to NO - 1 generate
        fwd_data_out_flat((NO - 1 - i + 1) * DATA_WIDTH - 1 downto (NO - 1 - i) * DATA_WIDTH) <= to_std_logic_vector(fwd_out_sf(i));
    end generate;
    gen_bo: for i in 0 to NI - 1 generate
        bwd_error_out_flat((NI - 1 - i + 1) * DATA_WIDTH - 1 downto (NI - 1 - i) * DATA_WIDTH) <= to_std_logic_vector(bwd_out_sf(i));
    end generate;
    gen_go: for i in 0 to NW - 1 generate
        grads_out_flat((NW - 1 - i + 1) * DATA_WIDTH - 1 downto (NW - 1 - i) * DATA_WIDTH) <= to_std_logic_vector(grads_sf(i));
    end generate;
end architecture wrap;

-- Neural network wrapper for XOR test (2→3→1)
-- Flattens output_data sfixed_bus_array to std_logic_vector.
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity nn_xor_wrap is
    port (
        clk : in std_logic;
        rst : in std_logic;
        load_weights   : in std_logic;
        start          : in std_logic;
        train_mode     : in std_logic;
        weights_loaded : out std_logic;
        ready          : out std_logic;
        done           : out std_logic;
        host_write_en  : in std_logic;
        host_addr      : in integer;
        host_data      : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_data_slv : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity nn_xor_wrap;

architecture wrap of nn_xor_wrap is
    constant LS : layer_config_array(0 to 1) := (3, 1);
    constant US : boolean_array(0 to 1) := (true, true);
    signal out_sf : sfixed_bus_array(0 to 0);
begin
    u_nn : entity work.neural_network
        generic map (
            NUM_INPUTS  => 2,
            NUM_LAYERS  => 2,
            LAYER_SIZES => LS,
            USE_SIGMOID => US
        )
        port map (
            clk => clk, rst => rst,
            load_weights   => load_weights,
            start          => start,
            train_mode     => train_mode,
            weights_loaded => weights_loaded,
            ready          => ready,
            done           => done,
            host_write_en  => host_write_en,
            host_addr      => host_addr,
            host_data      => host_data,
            output_data    => out_sf
        );

    output_data_slv <= to_std_logic_vector(out_sf(0));
end architecture wrap;
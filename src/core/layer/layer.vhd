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
        fwd_data_in  : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out std_logic_bus_array(0 to LAYER_SIZE - 1)(DATA_WIDTH - 1 downto 0);

        -- Backward Interface
        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in std_logic_bus_array(0 to LAYER_SIZE - 1)(DATA_WIDTH - 1 downto 0);
        
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);

        -- Weight Bank Interface (instead of direct weights_in)
        weight_load_en   : in  std_logic;
        weight_load_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_load_done : out std_logic;

        weight_save_en   : in  std_logic;
        weight_save_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_save_done : out std_logic;

        -- Gradient update (training)
        weight_update_en   : in  std_logic;
        weight_learn_rate  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        weight_update_done : out std_logic;
        
        -- Gradients Output (for external use if needed)
        grads_out : out std_logic_bus_array(0 to (NUM_INPUTS + 1) * LAYER_SIZE - 1)(DATA_WIDTH - 1 downto 0)
    );
end entity layer;

architecture rtl of layer is

    -- Weight count for this layer
    constant NUM_WEIGHTS : integer := (NUM_INPUTS + 1) * LAYER_SIZE;

    -- Internal Signals
    signal neuron_outputs : std_logic_bus_array(0 to LAYER_SIZE - 1)(DATA_WIDTH - 1 downto 0);
    
    -- Weights from weight bank (directly connected to neurons)
    signal weights_internal : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

    -- Gradients collected from neurons
    signal grads_internal : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);
    
    -- Input Error Accumulator (dL/dx sum from all neurons)
    type input_grad_array is array (0 to LAYER_SIZE - 1) of std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
    signal input_grads : input_grad_array;

    -- Helper to sum gradients for a specific input across all neurons
    function sum_input_grads(input_idx : integer; grads : input_grad_array) return std_logic_vector is
        variable sum : sfixed_bus := (others => '0');
        variable val : sfixed_bus;
    begin
        for i in 0 to LAYER_SIZE - 1 loop
            val := to_sfixed(grads(i)(input_idx), val);
            sum := resize(sum + val, sum);
        end loop;
        return to_std_logic_vector(sum);
    end function;

begin

    -- Instantiate Weight Bank
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

    -- Pass through control signals with pipeline delay
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

    -- Instantiate Neurons
    gen_neurons: for i in 0 to LAYER_SIZE - 1 generate
        -- Slice weights for this neuron: [w0..wN, bias]
        constant w_start : integer := i * (NUM_INPUTS + 1);
        
        signal n_weights : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        signal n_bias    : std_logic_vector(DATA_WIDTH - 1 downto 0);
        
        signal n_grad_weights : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        signal n_grad_bias    : std_logic_vector(DATA_WIDTH - 1 downto 0);
        
    begin
        -- Assign Weights from weight bank
        assign_w: for j in 0 to NUM_INPUTS - 1 generate
            n_weights(j) <= weights_internal(w_start + j);
        end generate;
        n_bias <= weights_internal(w_start + NUM_INPUTS);

        -- Assign Gradients to internal array (for weight bank update)
        assign_g: for j in 0 to NUM_INPUTS - 1 generate
            grads_internal(w_start + j) <= n_grad_weights(j);
        end generate;
        grads_internal(w_start + NUM_INPUTS) <= n_grad_bias;

        -- Neuron Instance
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

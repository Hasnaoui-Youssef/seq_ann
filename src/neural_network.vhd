library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity neural_network is
    generic(
        num_inputs : integer := 4;          -- Number of network inputs
        layer_sizes : layer_config_array;   -- Unconstrained array of layer sizes
        data_width : integer := 32;
        use_sigmoid : boolean := true
    );
    port (
        clk : in std_logic;
        rst : in std_logic;
        
        -- Control signals
        load_mode : in std_logic;           -- '1' = weight loading, '0' = inference
        
        -- Weight loading interface
        layer_select : in integer range 0 to 15;   -- Max 16 layers
        neuron_select : in integer range 0 to 15;  -- Max 16 neurons per layer
        weight_index : in integer range 0 to 15;   -- Max 16 inputs per neuron
        weight_data : in std_logic_vector(data_width - 1 downto 0);
        
        -- Inference interface
        inputs_i : in std_logic_bus_array(num_inputs - 1 downto 0)(data_width - 1 downto 0);
        
        -- Output (size depends on final layer)
        outputs_o : out std_logic_bus_array(0 to 15)(data_width - 1 downto 0);
        output_valid_o : out std_logic;
        
        -- Status
        ready_o : out std_logic
    );
end entity neural_network;

architecture rtl of neural_network is
    -- Number of layers is determined from array length
    constant num_layers : integer := layer_sizes'length;
    
    -- Helper function to get layer input size
    function get_layer_input_size(layer_idx : integer) return integer is
    begin
        if layer_idx = layer_sizes'left then
            return num_inputs;
        else
            return layer_sizes(layer_idx - 1);
        end if;
    end function;
    
    -- State machine
    type state_t is (IDLE, LOADING, READY, INFERENCE, OUTPUT);
    signal state : state_t := IDLE;
    
    -- Layer signals - we need max possible size
    type layer_output_array is array (0 to 15) of std_logic_bus_array(0 to 15)(data_width - 1 downto 0);
    signal layer_outputs : layer_output_array;
    
    -- Pipeline control
    signal inference_active : std_logic := '0';
    signal output_valid : std_logic := '0';
    
    -- Layer enable signals for weight loading
    type layer_load_enable_array is array (0 to 15) of std_logic;
    signal layer_load_enable : layer_load_enable_array;

begin
    -- ========================================================================
    -- State Machine
    -- ========================================================================
    process(clk, rst)
    begin
        if rst = '1' then
            state <= IDLE;
            output_valid <= '0';
            inference_active <= '0';
        elsif rising_edge(clk) then
            case state is
                when IDLE =>
                    output_valid <= '0';
                    if load_mode = '1' then
                        state <= LOADING;
                    else
                        state <= READY;
                    end if;
                    
                when LOADING =>
                    if load_mode = '0' then
                        state <= READY;
                    end if;
                    
                when READY =>
                    if load_mode = '1' then
                        state <= LOADING;
                    else
                        -- Start inference
                        inference_active <= '1';
                        state <= INFERENCE;
                    end if;
                    
                when INFERENCE =>
                    -- Wait for pipeline to complete
                    inference_active <= '0';
                    state <= OUTPUT;
                    
                when OUTPUT =>
                    output_valid <= '1';
                    state <= READY;
                    
                when others =>
                    state <= IDLE;
            end case;
        end if;
    end process;
    
    ready_o <= '1' when state = READY else '0';
    output_valid_o <= output_valid;
    
    -- ========================================================================
    -- Layer Load Enable Generation
    -- ========================================================================
    load_enable_gen : for i in 0 to 15 generate
        neuron_load_enable : process(load_mode, layer_select)
        begin
            if i < num_layers and load_mode = '1' and layer_select = i then
                layer_load_enable(i) <= '1';
            else
                layer_load_enable(i) <= '0';
            end if;
        end process;
    end generate;
    
    -- ========================================================================
    -- Dynamic Layer Instantiation
    -- ========================================================================
    layers_gen : for i in layer_sizes'range generate
        first_layer_gen : if i = layer_sizes'left generate
            layer_inst : entity work.layer
                generic map(
                    num_inputs => num_inputs,
                    num_outputs => layer_sizes(i),
                    data_width => data_width,
                    use_sigmoid => use_sigmoid
                )
                port map(
                    clk => clk,
                    inputs_i => inputs_i,
                    weights_matrix_i => (others => (others => (others => '0'))), -- Unused
                    load_enable => layer_load_enable(i),
                    neuron_select => neuron_select,
                    weight_data => weight_data,
                    weight_index => weight_index,
                    output_o => layer_outputs(i)(0 to layer_sizes(i) - 1)
                );
        end generate;
        
        other_layers_gen : if i /= layer_sizes'left generate
            layer_inst : entity work.layer
                generic map(
                    num_inputs => layer_sizes(i - 1),
                    num_outputs => layer_sizes(i),
                    data_width => data_width,
                    use_sigmoid => use_sigmoid
                )
                port map(
                    clk => clk,
                    inputs_i => layer_outputs(i - 1)(0 to layer_sizes(i - 1) - 1),
                    weights_matrix_i => (others => (others => (others => '0'))), -- Unused
                    load_enable => layer_load_enable(i),
                    neuron_select => neuron_select,
                    weight_data => weight_data,
                    weight_index => weight_index,
                    output_o => layer_outputs(i)(0 to layer_sizes(i) - 1)
                );
        end generate;
    end generate;
    
    -- ========================================================================
    -- Output Assignment
    -- ========================================================================
    output_assign : process(layer_outputs)
        variable last_layer_idx : integer;
    begin
        -- Default
        outputs_o <= (others => (others => '0'));
        
        -- Get last layer index
        last_layer_idx := layer_sizes'right;
        
        -- Assign output from final layer
        outputs_o(0 to layer_sizes(last_layer_idx) - 1) <= 
            layer_outputs(last_layer_idx)(0 to layer_sizes(last_layer_idx) - 1);
    end process;

end architecture rtl;

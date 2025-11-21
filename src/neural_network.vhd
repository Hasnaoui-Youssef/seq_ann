library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity neural_network is
    generic(
        num_inputs : integer := 4;          -- Number of network inputs
        num_layers : integer := 3;          -- Number of layers (hidden + output)
        -- Layer sizes: array of neuron counts per layer
        -- For example: (4, 3, 2) means layer 0 has 4 neurons, layer 1 has 3, layer 2 has 2
        layer_size_0 : integer := 4;
        layer_size_1 : integer := 3;
        layer_size_2 : integer := 2;
        layer_size_3 : integer := 1;        -- Unused if num_layers < 4
        data_width : integer := 32;
        use_sigmoid : boolean := true
    );
    port (
        clk : in std_logic;
        rst : in std_logic;
        
        -- Control signals
        load_mode : in std_logic;           -- '1' = weight loading, '0' = inference
        
        -- Weight loading interface
        layer_select : in integer range 0 to num_layers - 1;
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
    -- Helper function to get layer size
    function get_layer_size(layer_idx : integer) return integer is
    begin
        case layer_idx is
            when 0 => return layer_size_0;
            when 1 => return layer_size_1;
            when 2 => return layer_size_2;
            when 3 => return layer_size_3;
            when others => return 1;
        end case;
    end function;
    
    -- Helper function to get layer input size
    function get_layer_input_size(layer_idx : integer) return integer is
    begin
        if layer_idx = 0 then
            return num_inputs;
        else
            return get_layer_size(layer_idx - 1);
        end if;
    end function;
    
    -- State machine
    type state_t is (IDLE, LOADING, READY, INFERENCE, OUTPUT);
    signal state : state_t := IDLE;
    
    -- Layer signals
    type layer_output_array is array (0 to num_layers - 1) of std_logic_bus_array(0 to 15)(data_width - 1 downto 0);
    signal layer_outputs : layer_output_array;
    
    -- Pipeline control
    signal inference_active : std_logic := '0';
    signal output_valid : std_logic := '0';
    
    -- Layer enable signals for weight loading
    type layer_load_enable_array is array (0 to num_layers - 1) of std_logic;
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
                    -- With num_layers, we need num_layers + 1 clock cycles
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
    load_enable_gen : for i in 0 to num_layers - 1 generate
        layer_load_enable(i) <= '1' when (load_mode = '1' and layer_select = i) else '0';
    end generate;
    
    -- ========================================================================
    -- Layer Instantiation
    -- ========================================================================
    
    -- Layer 0 (first hidden layer)
    layer_0_gen : if num_layers >= 1 generate
        layer_0_inst : entity work.layer
            generic map(
                num_inputs => num_inputs,
                num_outputs => layer_size_0,
                data_width => data_width,
                use_sigmoid => use_sigmoid
            )
            port map(
                clk => clk,
                inputs_i => inputs_i,
                weights_matrix_i => (others => (others => (others => '0'))), -- Unused (internal storage)
                load_enable => layer_load_enable(0),
                neuron_select => neuron_select,
                weight_data => weight_data,
                weight_index => weight_index,
                output_o => layer_outputs(0)(0 to layer_size_0 - 1)
            );
    end generate;
    
    -- Layer 1
    layer_1_gen : if num_layers >= 2 generate
        layer_1_inst : entity work.layer
            generic map(
                num_inputs => layer_size_0,
                num_outputs => layer_size_1,
                data_width => data_width,
                use_sigmoid => use_sigmoid
            )
            port map(
                clk => clk,
                inputs_i => layer_outputs(0)(0 to layer_size_0 - 1),
                weights_matrix_i => (others => (others => (others => '0'))),
                load_enable => layer_load_enable(1),
                neuron_select => neuron_select,
                weight_data => weight_data,
                weight_index => weight_index,
                output_o => layer_outputs(1)(0 to layer_size_1 - 1)
            );
    end generate;
    
    -- Layer 2
    layer_2_gen : if num_layers >= 3 generate
        layer_2_inst : entity work.layer
            generic map(
                num_inputs => layer_size_1,
                num_outputs => layer_size_2,
                data_width => data_width,
                use_sigmoid => use_sigmoid
            )
            port map(
                clk => clk,
                inputs_i => layer_outputs(1)(0 to layer_size_1 - 1),
                weights_matrix_i => (others => (others => (others => '0'))),
                load_enable => layer_load_enable(2),
                neuron_select => neuron_select,
                weight_data => weight_data,
                weight_index => weight_index,
                output_o => layer_outputs(2)(0 to layer_size_2 - 1)
            );
    end generate;
    
    -- Layer 3 (optional)
    layer_3_gen : if num_layers >= 4 generate
        layer_3_inst : entity work.layer
            generic map(
                num_inputs => layer_size_2,
                num_outputs => layer_size_3,
                data_width => data_width,
                use_sigmoid => use_sigmoid
            )
            port map(
                clk => clk,
                inputs_i => layer_outputs(2)(0 to layer_size_2 - 1),
                weights_matrix_i => (others => (others => (others => '0'))),
                load_enable => layer_load_enable(3),
                neuron_select => neuron_select,
                weight_data => weight_data,
                weight_index => weight_index,
                output_o => layer_outputs(3)(0 to layer_size_3 - 1)
            );
    end generate;
    
    -- ========================================================================
    -- Output Assignment
    -- ========================================================================
    output_assign : process(layer_outputs)
    begin
        -- Default
        outputs_o <= (others => (others => '0'));
        
        -- Assign output from final layer
        case num_layers is
            when 1 =>
                outputs_o(0 to layer_size_0 - 1) <= layer_outputs(0)(0 to layer_size_0 - 1);
            when 2 =>
                outputs_o(0 to layer_size_1 - 1) <= layer_outputs(1)(0 to layer_size_1 - 1);
            when 3 =>
                outputs_o(0 to layer_size_2 - 1) <= layer_outputs(2)(0 to layer_size_2 - 1);
            when 4 =>
                outputs_o(0 to layer_size_3 - 1) <= layer_outputs(3)(0 to layer_size_3 - 1);
            when others =>
                outputs_o <= (others => (others => '0'));
        end case;
    end process;

end architecture rtl;

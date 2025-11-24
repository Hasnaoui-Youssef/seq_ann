library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;


entity layer is

    generic(
        layer_index : integer := 0;
        num_inputs : integer := 4;
        layer_size : integer := 4;
        use_sigmoid : boolean := true
    );
    port (
        clk : in std_logic;

        -- Data inputs
        -- Weight matrix (kept for compatibility, but neurons will use internal storage)


        -- Weight loading interface
        load_enable : in std_logic;
        neuron_select : in integer;
        weight_index : in integer;
        weight_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        
        -- Inference interface
        inputs_i : in std_logic_bus_array(0 to num_inputs - 1)(DATA_WIDTH - 1 downto 0);
        output_o : out std_logic_bus_array(0 to layer_size - 1)(DATA_WIDTH - 1 downto 0)
    );
end entity layer;

architecture rtl of layer is
    signal overflow_arr : std_logic_vector(0 to layer_size - 1);
    signal output_s : std_logic_bus_array(0 to layer_size - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));
    
    -- Internal signals for weight loading
    signal neuron_load_enable : std_logic_vector(0 to layer_size - 1);
    type load_enable_array is array (0 to layer_size - 1) of std_logic;

begin

    -- Generate load enable for each neuron
    load_enable_gen : for i in 0 to layer_size - 1 generate
        neuron_load_enable(i) <= '1' when (load_enable = '1' and neuron_select = i) else '0';
    end generate;

    neuron_gen : for i in 0 to layer_size - 1 generate
        neuron_inst: entity work.neuron
         generic map(
            layer_index => layer_index,
            neuron_index => i,
            num_inputs => num_inputs,
            use_sigmoid => use_sigmoid
        )
         port map(
            clk => clk,
            load_enable => neuron_load_enable(i),
            weight_index_i => weight_index,
            weight_data_i => weight_data,
            inputs_i => inputs_i,
            output_o => output_s(i)
        );
    end generate neuron_gen;



    reg_outputs: process(clk)
    begin
        if rising_edge(clk) then
            output_o <= output_s;
        end if;
    end process reg_outputs;


end architecture rtl;

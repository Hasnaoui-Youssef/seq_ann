library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;


entity layer is

    generic(
        num_inputs : integer := 4;
        num_outputs : integer := 4;
        data_width : integer := 16;
        use_sigmoid : boolean := true
    );
    port (
        clk : in std_logic;
        inputs_i : in std_logic_bus_array(num_inputs - 1 downto 0)(data_width - 1 downto 0);
        weights_matrix_i : in weights_matrix(0 to num_outputs - 1)(0 to num_inputs)(data_width - 1 downto 0);
        output_o : out std_logic_bus_array(0 to num_outputs - 1)(data_width - 1 downto 0)
    );
end entity layer;

architecture rtl of layer is
    signal overflow_arr : std_logic_vector(0 to num_outputs - 1);
    signal output_s : std_logic_bus_array(0 to num_outputs - 1)(data_width - 1 downto 0) := (others => (others => '0'));
    signal inputs_s : std_logic_bus_array(num_inputs - 1 downto 0)(data_width - 1 downto 0):= (others => (others => '0'));
begin

    neuron_gen : for i in 0 to num_outputs - 1 generate
        neuron_inst: entity work.neuron
         generic map(
            num_inputs => num_inputs,
            data_width => data_width,
            use_sigmoid => use_sigmoid
        )
         port map(
            inputs_i => inputs_s,
            weights_i => weights_matrix_i(i),
            output_o => output_s(i),
            overflow_o => overflow_arr(i)
        );
    end generate neuron_gen;

    reg_inputs: process(clk)
    begin
        if rising_edge(clk) then
            inputs_s <= inputs_i;
        end if;
    end process reg_inputs;

    reg_outputs: process(clk)
    begin
        if rising_edge(clk) then
            output_o <= output_s;
        end if;
    end process reg_outputs;


end architecture rtl;

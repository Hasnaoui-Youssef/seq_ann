library ieee;
use ieee.std_logic_1164.all;
use work.types.all;


entity acc is
    generic (
        size : integer := 4;
        data_width : integer := 4;
        signed_acc : boolean := false
    );
    port (
        inputs : in std_logic_bus_array(size - 1 downto 0)(data_width - 1 downto 0);
        sum_o : out std_logic_vector(size + data_width - 1 downto 0);
        overflow_o : out std_logic
    );
end entity acc;

architecture rtl of acc is
    signal carry_sigs : std_logic_vector(size-1 downto 0);
    signal result_sigs : std_logic_bus_array(size-1 downto 0)(size + data_width - 1 downto 0);
    signal input_resize : std_logic_bus_array(size - 1 downto 0) (size + data_width - 1 downto 0);

begin

    -- Resizing n bits to m bits with sign =>
    -- MSB must remain the same to keep track of the sign
    -- Fill with zeros from MSB - 1 upuntil the original size + 1 (since the last element of the original signal is the sign bit)
    -- Copy the ramining elements in place
    input_assign : for i in 0 to size - 1 generate
            --input_resize(i)(size + data_width - 1) <= inputs(i)(data_width - 1);
        sign_gen : if signed_acc = true generate
            input_resize(i)(size + data_width - 1 downto data_width ) <= (others => inputs(i)(data_width - 1));
        else generate
            input_resize(i)(size + data_width - 1 downto data_width ) <= (others => '0');
        end generate;
            input_resize(i)(data_width - 1 downto 0) <= inputs(i)(data_width - 1 downto 0);
    end generate input_assign;
    result_sigs(0) <= input_resize(0);
    carry_sigs(0) <= '0';
    n_adders_gen : for i in 1 to size - 1 generate
        n_adder_inst: entity work.n_bit_adder
        generic map(
                       size => data_width + data_width
                   )
        port map(
            a => result_sigs(i - 1),
            b => input_resize(i),
            carry_i => carry_sigs(i - 1),
            sum_o => result_sigs(i),
            overflow_o => carry_sigs(i)
        );
    end generate n_adders_gen;
    sum_o <= result_sigs(size - 1);
    overflow_o <= carry_sigs(size - 1);

end architecture rtl;

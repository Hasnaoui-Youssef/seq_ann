library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func is
    generic(
        input_width : integer := 48;
        output_width : integer := 32
    );
    port(
        input_i : in std_logic_vector(input_width / 2 - 1 downto - (input_width / 2));
        output_o : out sfixed(output_width / 2 - 1 downto - (output_width / 2)) := (others => '0')
    );
end entity activation_func;

architecture relu of activation_func is
begin
    output_o <= to_sfixed(input_i, output_o) when signed(input_i) >= 0 else to_sfixed_a(0);
end architecture relu;

architecture sigmoid of activation_func is
    signal input_sfixed : sfixed(input_width / 2 - 1 downto - (input_width / 2));
    signal lut_index : integer range 0 to LUT_SIZE - 1;
    signal sigmoid_value : real;

begin
    -- Convert input to sfixed
    input_sfixed <= to_sfixed(input_i, input_sfixed);

    -- Extract middle bits for LUT indexing (8 bits from input)
    -- Using bits 3 downto -4
    lut_index <= to_integer(unsigned(std_logic_vector(input_sfixed(3 downto -4))));

    -- Lookup sigmoid value from LUT
    sigmoid_value <= SIGMOID_LUT(lut_index);

    -- Convert to output fixed-point format
    output_o <= to_sfixed(sigmoid_value, output_o'high, output_o'low);

end architecture sigmoid;

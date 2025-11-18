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
        input_i : in std_logic_vector(input_width - 1 downto 0);
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
    signal input_real : real;
    signal clipped : real;
    signal normalized : real;

begin
    -- Convert input to sfixed
    input_sfixed <= to_sfixed(input_i, input_sfixed);
    input_real <= to_real(input_sfixed);
    clipped <= INPUT_MAX when input_real > INPUT_MAX else INPUT_MIN when input_real < INPUT_MIN else input_real;
    normalized <= ((clipped - INPUT_MIN)/(INPUT_MAX - INPUT_MIN));

    -- Extract middle bits for LUT indexing (10 bits from input)
    -- Using bits 4 downto -5
    lut_index <= 0 when (normalized  < 0.0)
                 else (LUT_SIZE - 1) when (normalized > 1.0)
                 else integer(normalized * real(LUT_SIZE - 1));


    -- Lookup sigmoid value from LUT
    sigmoid_value <= SIGMOID_LUT(lut_index);

    -- Convert to output fixed-point format
    output_o <= to_sfixed(sigmoid_value, output_o'high, output_o'low);

end architecture sigmoid;

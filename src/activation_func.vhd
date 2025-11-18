library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;


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

begin



end architecture sigmoid;

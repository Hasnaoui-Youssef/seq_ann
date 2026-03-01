library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use work.types.all;

entity mult is
    port (
        a_i : in sfixed_bus;
        b_i : in sfixed_bus;
        product_o : out sfixed_bus
    );
end entity mult;

architecture rtl of mult is
begin
    product_o <= resize(a_i * b_i, INT_BITS - 1, -FRAC_BITS);
end architecture rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mult is
    port (
        a_i : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        b_i : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        product_o : out std_logic_vector(2 * DATA_WIDTH - 1 downto 0)
    );
end entity mult;

architecture rtl of mult is
    signal a_signed : signed(DATA_WIDTH - 1 downto 0);
    signal b_signed : signed(DATA_WIDTH - 1 downto 0);
    signal product_signed : signed(2 * DATA_WIDTH - 1 downto 0);
begin
    -- Convert inputs to signed
    a_signed <= signed(a_i);
    b_signed <= signed(b_i);

    -- Perform signed multiplication
    product_signed <= a_signed * b_signed;

    -- Convert back to std_logic_vector
    product_o <= std_logic_vector(product_signed);

end architecture rtl;

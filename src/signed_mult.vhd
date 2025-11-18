library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mult is
    generic (
        data_width : integer := 16
    );
    port (
        a_i : in std_logic_vector(data_width - 1 downto 0);
        b_i : in std_logic_vector(data_width - 1 downto 0);
        product_o : out std_logic_vector(2 * data_width - 1 downto 0)
    );
end entity mult;

architecture rtl of mult is
    signal a_signed : signed(data_width - 1 downto 0);
    signal b_signed : signed(data_width - 1 downto 0);
    signal product_signed : signed(2 * data_width - 1 downto 0);
begin
    -- Convert inputs to signed
    a_signed <= signed(a_i);
    b_signed <= signed(b_i);

    -- Perform signed multiplication
    product_signed <= a_signed * b_signed;

    -- Convert back to std_logic_vector
    product_o <= std_logic_vector(product_signed);

end architecture rtl;

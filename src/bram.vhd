library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

entity bram is
    generic(
        DATA_WIDTH : integer := 16;
        ADDR_WIDTH : integer := 16
    );
    port(
        clka : in std_logic;
        clkb : in std_logic;
        ena : in std_logic;
        enb : in std_logic;
        wea : in std_logic;
        web : in std_logic;
        addra : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
        addrb : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
        dia : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        dib : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        doa : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        dob : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end bram;

architecture syn of bram is
type ram_type is array (2**ADDR_WIDTH - 1 downto 0) of std_logic_vector(DATA_WIDTH - 1 downto 0);
shared variable RAM : ram_type;
begin
process(CLKA)
    begin
    if CLKA'event and CLKA = '1' then
        if ENA = '1' then
            DOA <= RAM(to_integer(unsigned(ADDRA)));
            if WEA = '1' then
                RAM(to_integer(unsigned(ADDRA))) := DIA;
            end if;
        end if;
    end if;
end process;

process(CLKB)
    begin
    if CLKB'event and CLKB = '1' then
        if ENB = '1' then
            DOB <= RAM(to_integer(unsigned(ADDRB)));
            if WEB = '1' then
                RAM(to_integer(unsigned(ADDRB))) := DIB;
            end if;
        end if;
    end if;
end process;

end syn;
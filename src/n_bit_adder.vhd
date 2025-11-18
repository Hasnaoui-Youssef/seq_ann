library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity n_bit_adder is
    generic(
        SIZE : integer
    );
    port (
             a : in std_logic_vector(SIZE - 1 downto 0);
             b : in std_logic_vector(SIZE - 1 downto 0);
             carry_i : in std_logic;
             sum_o : out std_logic_vector(SIZE - 1 downto 0);
             overflow_o : out std_logic
         );
end entity n_bit_adder;

architecture rtl of n_bit_adder is
    signal carry_sigs : std_logic_vector(SIZE downto 0);

begin
    carry_sigs(0) <= carry_i;
    adders_gen : for i in 0 to SIZE - 1 generate
            full_adder_inst: entity work.full_adder
             port map(
                bit1_i => a(i),
                bit2_i => b(i),
                carry_i => carry_sigs(i),
                sum_o => sum_o(i),
                carry_o => carry_sigs(i + 1)
            );
    end generate adders_gen;
    overflow_o <= '0' when SIZE=1 else carry_sigs(SIZE - 1) xor carry_sigs(SIZE);
end architecture rtl;

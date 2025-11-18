library ieee;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity full_adder is
    port (
        bit1_i : in std_logic;
        bit2_i : in std_logic;
        carry_i : in std_logic;
        sum_o : out std_logic;
        carry_o : out std_logic
    );
end entity full_adder;

architecture rtl of full_adder is
    signal pq_sum : std_logic;
    signal pq_carry : std_logic;
    signal upper_carry : std_logic;

begin
    carry_adder : entity work.half_adder
     port map(
        bit1_i => carry_i,
        bit2_i => pq_sum,
        sum_o => sum_o,
        carry_o => upper_carry
    );
    pq_adder : entity work.half_adder
     port map(
        bit1_i => bit1_i,
        bit2_i => bit2_i,
        sum_o => pq_sum,
        carry_o => pq_carry
    );

    carry_o <= pq_carry or upper_carry;
end architecture rtl;

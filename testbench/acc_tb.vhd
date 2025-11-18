library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.types.all;


entity acc_tb is
end entity acc_tb;

architecture testbench of acc_tb is
    constant size : integer := 4;
    constant data_width : integer := 4;
    signal tb_a : std_logic_bus_array(size - 1 downto 0)(data_width - 1 downto 0);
    signal tb_sum_unsigned : std_logic_vector(size + data_width - 1 downto 0);
    signal tb_sum_signed : std_logic_vector(size + data_width - 1 downto 0);
    signal tb_overflow_signed : std_logic;
    signal tb_overflow_unsigned : std_logic;

begin

    signed_acc: entity work.acc
     generic map(
        size,
        data_width,
        signed_acc => true
    )
     port map(
        inputs => tb_a,
        sum_o => tb_sum_signed,
        overflow_o => tb_overflow_signed
    );
    unsigned_acc: entity work.acc
     generic map(
        size,
        data_width,
        signed_acc => false
    )
     port map(
        inputs => tb_a,
        sum_o => tb_sum_unsigned,
        overflow_o => tb_overflow_unsigned
    );

    sim_p : process
    begin
        tb_a <= ("0011", "0001", "0100", "0010");
        wait for 10 ns;
        assert (unsigned(tb_sum_unsigned) = 10 and signed(tb_sum_signed) = 10) report "Test Case 1 Failed" severity error;
        tb_a <= ("1111", "1111", "1111", "1111");
        wait for 10 ns;
        assert (unsigned(tb_sum_unsigned) = 60 and signed(tb_sum_signed) = -4) report "Test Case 2 Failed" severity error;
        tb_a <= ("0000", "0000", "0000", "0000");
        wait for 10 ns;
        assert (unsigned(tb_sum_unsigned) = 0 and signed(tb_sum_signed) = 0) report "Test Case 3 Failed" severity error;
        tb_a <= ("0001", "0010", "0100", "1000");
        wait for 10 ns;
        assert (unsigned(tb_sum_unsigned) = 15 and signed(tb_sum_signed) = -1) report "Test Case 4 Failed" severity error;
        report "All 4-bit 4 number Accumulator test cases completed." severity note;
        wait; -- Hold simulation indefinitely

    end process sim_p;



end architecture testbench;


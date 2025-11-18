library ieee;
use ieee.std_logic_1164.all;


entity n_bit_adder_tb is
end entity n_bit_adder_tb;

architecture testbench of n_bit_adder_tb is
    constant size : integer := 4;
    signal tb_a : std_logic_vector(size - 1 downto 0);
    signal tb_b : std_logic_vector(size - 1 downto 0);
    signal tb_sum : std_logic_vector(size - 1 downto 0);
    signal tb_carry_out : std_logic;
    signal tb_carry_in : std_logic := '0';

begin

    n_bit_adder_inst: entity work.n_bit_adder
     generic map(
        SIZE => size
    )
     port map(
        a => tb_a,
        b => tb_b,
        carry_i => tb_carry_in,
        sum_o => tb_sum,
        overflow_o => tb_carry_out
    );

    sim_p : process
    begin
        tb_a <= "0011"; tb_b <= "0001"; tb_carry_in <= '0';
        wait for 10 ns;
        assert (tb_sum = "0100" and tb_carry_out = '0') report "Test Case 1 Failed" severity error;
        tb_a <= "1111"; tb_b <= "1111"; tb_carry_in <= '1';
        wait for 10 ns;
        assert (tb_sum = "1111" and tb_carry_out = '1') report "Test Case 2 Failed" severity error;
        tb_a <= "1010"; tb_b <= "0101"; tb_carry_in <= '1';
        wait for 10 ns;
        assert (tb_sum = "0000" and tb_carry_out = '1') report "Test Case 3 Failed" severity error;
        tb_a <= "1111"; tb_b <= "0001"; tb_carry_in <= '1';
        wait for 10 ns;
        assert (tb_sum = "0001" and tb_carry_out = '1') report "Test Case 4 Failed" severity error;
        report "All 4-bit adder test cases completed." severity note;
        wait; -- Hold simulation indefinitely

    end process sim_p;



end architecture testbench;

library ieee;
use ieee.std_logic_1164.all;


entity full_adder_tb is
end entity full_adder_tb;

architecture testbench of full_adder_tb is
    signal tb_a, tb_b, tb_cin : std_logic := '0';
    signal tb_sum, tb_cout : std_logic;

begin
    full_adder_inst: entity work.full_adder
    port map (
        bit1_i  => tb_a,
        bit2_i => tb_b,
        carry_i => tb_cin,
        sum_o => tb_sum,
        carry_o => tb_cout
    );

    sim_p : process
    begin
        tb_a <= '0'; tb_b <= '0'; tb_cin <= '0';
        wait for 10 ns;
        assert (tb_sum = '0' and tb_cout = '0') report "Test Case 1 Failed" severity error;

        -- Test Case 2: 0 + 0 + 1 = 1 (sum), 0 (cout)
        tb_a <= '0'; tb_b <= '0'; tb_cin <= '1';
        wait for 10 ns;
        assert (tb_sum = '1' and tb_cout = '0') report "Test Case 2 Failed" severity error;

        -- Test Case 3: 0 + 1 + 0 = 1 (sum), 0 (cout)
        tb_a <= '0'; tb_b <= '1'; tb_cin <= '0';
        wait for 10 ns;
        assert (tb_sum = '1' and tb_cout = '0') report "Test Case 3 Failed" severity error;

        -- Test Case 4: 0 + 1 + 1 = 0 (sum), 1 (cout)
        tb_a <= '0'; tb_b <= '1'; tb_cin <= '1';
        wait for 10 ns;
        assert (tb_sum = '0' and tb_cout = '1') report "Test Case 4 Failed" severity error;

        -- Test Case 5: 1 + 0 + 0 = 1 (sum), 0 (cout)
        tb_a <= '1'; tb_b <= '0'; tb_cin <= '0';
        wait for 10 ns;
        assert (tb_sum = '1' and tb_cout = '0') report "Test Case 5 Failed" severity error;

        -- Test Case 6: 1 + 0 + 1 = 0 (sum), 1 (cout)
        tb_a <= '1'; tb_b <= '0'; tb_cin <= '1';
        wait for 10 ns;
        assert (tb_sum = '0' and tb_cout = '1') report "Test Case 6 Failed" severity error;

        -- Test Case 7: 1 + 1 + 0 = 0 (sum), 1 (cout)
        tb_a <= '1'; tb_b <= '1'; tb_cin <= '0';
        wait for 10 ns;
        assert (tb_sum = '0' and tb_cout = '1') report "Test Case 7 Failed" severity error;

        -- Test Case 8: 1 + 1 + 1 = 1 (sum), 1 (cout)
        tb_a <= '1'; tb_b <= '1'; tb_cin <= '1';
        wait for 10 ns;
        assert (tb_sum = '1' and tb_cout = '1') report "Test Case 8 Failed" severity error;

        report "All full adder test cases completed." severity note;
        wait; -- Hold simulation indefinitely

    end process sim_p;



end architecture testbench;

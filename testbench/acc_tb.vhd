library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity acc_tb is
end entity acc_tb;

architecture behavior of acc_tb is
    constant SIZE : integer := 2;
    
    signal inputs : std_logic_bus_array(SIZE-1 downto 0)(DATA_WIDTH-1 downto 0);
    signal sum : std_logic_vector(DATA_WIDTH + SIZE - 1 downto 0);
    signal overflow : std_logic;
    
begin
    uut: entity work.acc
        generic map(
            size => SIZE,
            signed_acc => true
        )
        port map(
            inputs => inputs,
            sum_o => sum,
            overflow_o => overflow
        );

    process
    begin
        report "Starting ACC Testbench...";
        report "DATA_WIDTH = " & integer'image(DATA_WIDTH);
        
        -- Test 1: 1 + 1 = 2
        inputs(0) <= std_logic_vector(to_signed(1, DATA_WIDTH));
        inputs(1) <= std_logic_vector(to_signed(1, DATA_WIDTH));
        wait for 10 ns;
        report "1+1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = 2 report "Fail 1+1" severity error;
        
        -- Test 2: -1 + -1 = -2
        inputs(0) <= std_logic_vector(to_signed(-1, DATA_WIDTH));
        inputs(1) <= std_logic_vector(to_signed(-1, DATA_WIDTH));
        wait for 10 ns;
        report "-1+-1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = -2 report "Fail -1+-1" severity error;
        
        -- Test 3: Large positive + 1
        inputs(0) <= std_logic_vector(to_signed(1000, DATA_WIDTH));
        inputs(1) <= std_logic_vector(to_signed(1, DATA_WIDTH));
        wait for 10 ns;
        report "1000+1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = 1001 report "Fail 1000+1" severity error;
        
        -- Test 4: Large negative + -1
        inputs(0) <= std_logic_vector(to_signed(-1000, DATA_WIDTH));
        inputs(1) <= std_logic_vector(to_signed(-1, DATA_WIDTH));
        wait for 10 ns;
        report "-1000+-1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = -1001 report "Fail -1000+-1" severity error;
        
        report "ACC Testbench Complete!";
        wait;
    end process;
end architecture;

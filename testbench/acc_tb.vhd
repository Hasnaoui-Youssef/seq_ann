library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity acc_tb is
end entity acc_tb;

architecture behavior of acc_tb is
    constant SIZE : integer := 2;
    constant WIDTH : integer := 8;
    
    signal inputs : std_logic_bus_array(SIZE-1 downto 0)(WIDTH-1 downto 0);
    signal sum : std_logic_vector(WIDTH + SIZE - 1 downto 0); -- Match acc output width
    signal overflow : std_logic;
    
begin
    uut: entity work.acc
        generic map(
            size => SIZE,
            data_width => WIDTH,
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
        
        -- Test 1: 1 + 1 = 2
        inputs(0) <= x"01";
        inputs(1) <= x"01";
        wait for 10 ns;
        report "1+1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = 2 report "Fail 1+1" severity error;
        
        -- Test 2: -1 + -1 = -2
        inputs(0) <= x"FF"; -- -1
        inputs(1) <= x"FF"; -- -1
        wait for 10 ns;
        report "-1+-1=" & integer'image(to_integer(signed(sum)));
        assert to_integer(signed(sum)) = -2 report "Fail -1+-1" severity error;
        
        -- Test 3: Max Pos + 1
        inputs(0) <= x"7F"; -- 127
        inputs(1) <= x"01"; -- 1
        wait for 10 ns;
        report "127+1=" & integer'image(to_integer(signed(sum)));
        -- Sum width is 9 bits. 127+1 = 128.
        -- 128 in 9 bits is 0010000000? No. 0 1000 0000.
        -- 127 is 0 0111 1111.
        -- 1 is 0 0000 0001.
        -- Sum 0 1000 0000.
        
        -- Test 4: Max Neg + -1
        inputs(0) <= x"80"; -- -128
        inputs(1) <= x"FF"; -- -1
        wait for 10 ns;
        report "-128+-1=" & integer'image(to_integer(signed(sum)));
        
        wait;
    end process;
end architecture;

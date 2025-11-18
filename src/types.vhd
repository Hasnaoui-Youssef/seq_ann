library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;

package types is

    constant width : integer := 16;

    subtype std_logic_bus is std_logic_vector(width - 1 downto 0);
    type std_logic_bus_array is array (integer range<>) of std_logic_vector;

    constant int_s : integer := 16;
    constant frac_s : integer := 16;

    subtype sfixed_bus is sfixed(int_s - 1 downto -frac_s); -- (15 -> -16)
    type sfixed_bus_array is array (integer range<>) of sfixed_bus;

    type real_array is array (integer range<>) of real;

    function to_sfixed_a(arg: integer) return unresolved_sfixed;
    function to_sfixed_a(arg: real) return unresolved_sfixed;
    function to_real(arg: sfixed_bus_array) return real_array;

end package types;

package body types is
    function to_sfixed_a(arg: integer) return unresolved_sfixed is
        variable result : unresolved_sfixed(int_s - 1 downto -frac_s);
    begin
        result := to_sfixed(
                arg => arg,
                left_index => int_s - 1,
                right_index => -frac_s);
        return result;
    end function to_sfixed_a;
    function to_sfixed_a(arg: real) return unresolved_sfixed is
        variable result : unresolved_sfixed(int_s - 1 downto -frac_s);
    begin
        result := to_sfixed(
                arg => arg,
                left_index => int_s - 1,
                right_index => -frac_s);
        return result;
    end function to_sfixed_a;

    function to_real(arg: sfixed_bus_array) return real_array is
        variable result : real_array(arg'range);
    begin
        for i in arg'range loop
            result(i) := to_real(arg => arg(i));
        end loop;
        return result;
    end function to_real;
end package body types;

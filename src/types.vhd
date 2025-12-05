library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;

package types is

    -- Global Fixed-Point Configuration
    constant DATA_WIDTH : integer := 32;
    constant FRAC_BITS : integer := 16;
    
    -- Derived constants
    constant INT_BITS : integer := DATA_WIDTH - FRAC_BITS;

    subtype std_logic_bus is std_logic_vector(DATA_WIDTH - 1 downto 0);
    type std_logic_bus_array is array (integer range<>) of std_logic_vector; -- for neuron weights and inputs
    type weights_matrix is array (integer range<>) of std_logic_bus_array; -- for layer weights
    -- Unconstrained integer array for layer sizes (avoiding conflict with VHDL-2008 integer_vector)
    type layer_config_array is array (natural range <>) of integer;
    type weights_tensor is array (integer range<>) of weights_matrix; -- for network weights

    type real_array is array (integer range<>) of real;

    function to_sfixed_a(arg: integer) return unresolved_sfixed;
    function to_sfixed_a(arg: real) return unresolved_sfixed;
    -- function to_real(arg: sfixed_bus_array) return real_array; -- Removed as sfixed_bus_array is gone

end package types;

package body types is
    function to_sfixed_a(arg: integer) return unresolved_sfixed is
        variable result : unresolved_sfixed(INT_BITS - 1 downto -FRAC_BITS);
    begin
        result := to_sfixed(
                arg => arg,
                left_index => INT_BITS - 1,
                right_index => -FRAC_BITS);
        return result;
    end function to_sfixed_a;
    function to_sfixed_a(arg: real) return unresolved_sfixed is
        variable result : unresolved_sfixed(INT_BITS - 1 downto -FRAC_BITS);
    begin
        result := to_sfixed(
                arg => arg,
                left_index => INT_BITS - 1,
                right_index => -FRAC_BITS);
        return result;
    end function to_sfixed_a;

end package body types;

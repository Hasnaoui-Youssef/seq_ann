library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

package pkg_layer is

    -- ========================================================================
    -- Control Signals (Records)
    -- ========================================================================
    
    type layer_control_t is record
        valid : std_logic;
        last  : std_logic; -- End of sequence/batch
    end record;

    -- ========================================================================
    -- Data Types (Aliases for clarity)
    -- ========================================================================
    
    -- Data is passed as std_logic_bus_array (defined in types.vhd)
    -- This allows unconstrained ports: 
    -- data_in : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);

end package pkg_layer;

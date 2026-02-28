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

end package pkg_layer;

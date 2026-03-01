library IEEE;
use IEEE.std_logic_1164.all;
use work.types.all;
use work.pkg_layer.all;

-- Flatten Layer: Reshapes multi-dimensional feature maps to 1D vector
--
-- Pure routing — no computation, no learnable parameters.
-- Since our data buses are already 1D arrays, this is a pass-through
-- that serves as a semantic boundary between spatial and dense layers.

entity flatten_layer is
    generic (
        TOTAL_SIZE : integer := 784  -- Total number of elements (C * H * W)
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        fwd_ctrl_in  : in layer_control_t;
        fwd_data_in  : in sfixed_bus_array(0 to TOTAL_SIZE - 1);
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out sfixed_bus_array(0 to TOTAL_SIZE - 1);

        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in sfixed_bus_array(0 to TOTAL_SIZE - 1);
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out sfixed_bus_array(0 to TOTAL_SIZE - 1)
    );
end entity flatten_layer;

architecture rtl of flatten_layer is
begin

    fwd_data_out <= fwd_data_in;
    bwd_error_out <= bwd_error_in;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                fwd_ctrl_out.valid <= '0';
                fwd_ctrl_out.last <= '0';
                bwd_ctrl_out.valid <= '0';
                bwd_ctrl_out.last <= '0';
            else
                fwd_ctrl_out <= fwd_ctrl_in;
                bwd_ctrl_out <= bwd_ctrl_in;
            end if;
        end if;
    end process;

end architecture rtl;

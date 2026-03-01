library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;
use work.layer_interface_pkg.all;

-- AvgPool2D Layer: Spatial downsampling via averaging
--
-- No learnable parameters. Backward pass distributes gradient equally.

entity avgpool_layer is
    generic (
        C     : integer := 1;
        H_IN  : integer := 28;
        W_IN  : integer := 28;
        POOL_H : integer := 2;
        POOL_W : integer := 2;
        STRIDE_H : integer := 2;
        STRIDE_W : integer := 2
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        fwd_ctrl_in  : in layer_control_t;
        fwd_data_in  : in sfixed_bus_array(0 to C * H_IN * W_IN - 1);
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out sfixed_bus_array(0 to C * ((H_IN - POOL_H)/STRIDE_H + 1) * ((W_IN - POOL_W)/STRIDE_W + 1) - 1);

        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in sfixed_bus_array(0 to C * ((H_IN - POOL_H)/STRIDE_H + 1) * ((W_IN - POOL_W)/STRIDE_W + 1) - 1);
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out sfixed_bus_array(0 to C * H_IN * W_IN - 1)
    );
end entity avgpool_layer;

architecture rtl of avgpool_layer is

    constant H_OUT : integer := (H_IN - POOL_H) / STRIDE_H + 1;
    constant W_OUT : integer := (W_IN - POOL_W) / STRIDE_W + 1;
    constant POOL_SIZE : integer := POOL_H * POOL_W;
    constant OUTPUT_SIZE : integer := C * H_OUT * W_OUT;

    signal output_reg : sfixed_bus_array(0 to OUTPUT_SIZE - 1)
        := (others => (others => '0'));

    -- Precompute 1/pool_size as fixed-point
    constant INV_POOL : sfixed_bus := to_sfixed(1.0 / real(POOL_SIZE), INT_BITS - 1, -FRAC_BITS);

begin

    fwd_data_out <= output_reg;

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

    -- Forward: compute average in each pool window (combinational)
    process(fwd_data_in)
        variable acc : sfixed_bus;
        variable in_idx : integer;
    begin
        for ch in 0 to C - 1 loop
            for oh in 0 to H_OUT - 1 loop
                for ow in 0 to W_OUT - 1 loop
                    acc := (others => '0');
                    for ph in 0 to POOL_H - 1 loop
                        for pw in 0 to POOL_W - 1 loop
                            in_idx := ch * H_IN * W_IN +
                                      (oh * STRIDE_H + ph) * W_IN +
                                      (ow * STRIDE_W + pw);
                            acc := resize(acc + fwd_data_in(in_idx),
                                         INT_BITS - 1, -FRAC_BITS);
                        end loop;
                    end loop;
                    output_reg(ch * H_OUT * W_OUT + oh * W_OUT + ow) <=
                        resize(acc * INV_POOL, INT_BITS - 1, -FRAC_BITS);
                end loop;
            end loop;
        end loop;
    end process;

    -- Backward: distribute gradient equally to all positions in pool window
    process(clk)
        variable grad_sf : sfixed_bus;
        variable distributed : sfixed_bus;
        variable in_idx : integer;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                bwd_error_out <= (others => (others => '0'));
            elsif bwd_ctrl_in.valid = '1' then
                bwd_error_out <= (others => (others => '0'));
                for ch in 0 to C - 1 loop
                    for oh in 0 to H_OUT - 1 loop
                        for ow in 0 to W_OUT - 1 loop
                            grad_sf := bwd_error_in(ch * H_OUT * W_OUT + oh * W_OUT + ow);
                            distributed := resize(grad_sf * INV_POOL, INT_BITS - 1, -FRAC_BITS);
                            for ph in 0 to POOL_H - 1 loop
                                for pw in 0 to POOL_W - 1 loop
                                    in_idx := ch * H_IN * W_IN +
                                              (oh * STRIDE_H + ph) * W_IN +
                                              (ow * STRIDE_W + pw);
                                    bwd_error_out(in_idx) <= distributed;
                                end loop;
                            end loop;
                        end loop;
                    end loop;
                end loop;
            end if;
        end if;
    end process;

end architecture rtl;

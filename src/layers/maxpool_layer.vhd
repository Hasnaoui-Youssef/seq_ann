library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;
use work.layer_interface_pkg.all;

-- MaxPool2D Layer: Spatial downsampling via max operation
--
-- Tracks indices of maximum values for backward pass routing.
-- No learnable parameters (no weight bank needed).

entity maxpool_layer is
    generic (
        C     : integer := 1;    -- Number of channels
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

        -- Forward Interface
        fwd_ctrl_in  : in layer_control_t;
        fwd_data_in  : in std_logic_bus_array(0 to C * H_IN * W_IN - 1)(DATA_WIDTH - 1 downto 0);
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out std_logic_bus_array(0 to C * ((H_IN - POOL_H)/STRIDE_H + 1) * ((W_IN - POOL_W)/STRIDE_W + 1) - 1)(DATA_WIDTH - 1 downto 0);

        -- Backward Interface
        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in std_logic_bus_array(0 to C * ((H_IN - POOL_H)/STRIDE_H + 1) * ((W_IN - POOL_W)/STRIDE_W + 1) - 1)(DATA_WIDTH - 1 downto 0);
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out std_logic_bus_array(0 to C * H_IN * W_IN - 1)(DATA_WIDTH - 1 downto 0)
    );
end entity maxpool_layer;

architecture rtl of maxpool_layer is

    constant H_OUT : integer := (H_IN - POOL_H) / STRIDE_H + 1;
    constant W_OUT : integer := (W_IN - POOL_W) / STRIDE_W + 1;
    constant INPUT_SIZE  : integer := C * H_IN * W_IN;
    constant OUTPUT_SIZE : integer := C * H_OUT * W_OUT;

    -- Max index storage for backward routing
    type index_array_t is array (0 to OUTPUT_SIZE - 1) of integer range 0 to INPUT_SIZE - 1;
    signal max_indices : index_array_t := (others => 0);

    signal output_reg : std_logic_bus_array(0 to OUTPUT_SIZE - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

begin

    fwd_data_out <= output_reg;

    -- Control propagation
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

    -- Forward pass: max pooling with index tracking (combinational)
    process(fwd_data_in)
        variable max_val : sfixed_bus;
        variable cur_val : sfixed_bus;
        variable max_idx : integer;
        variable in_idx  : integer;
        variable out_idx : integer;
    begin
        for ch in 0 to C - 1 loop
            for oh in 0 to H_OUT - 1 loop
                for ow in 0 to W_OUT - 1 loop
                    out_idx := ch * H_OUT * W_OUT + oh * W_OUT + ow;
                    -- Initialize with first element in pool window
                    in_idx := ch * H_IN * W_IN + (oh * STRIDE_H) * W_IN + (ow * STRIDE_W);
                    max_val := to_sfixed(fwd_data_in(in_idx), INT_BITS - 1, -FRAC_BITS);
                    max_idx := in_idx;

                    -- Find maximum in pool window
                    for ph in 0 to POOL_H - 1 loop
                        for pw in 0 to POOL_W - 1 loop
                            in_idx := ch * H_IN * W_IN +
                                      (oh * STRIDE_H + ph) * W_IN +
                                      (ow * STRIDE_W + pw);
                            cur_val := to_sfixed(fwd_data_in(in_idx), INT_BITS - 1, -FRAC_BITS);
                            if cur_val > max_val then
                                max_val := cur_val;
                                max_idx := in_idx;
                            end if;
                        end loop;
                    end loop;

                    output_reg(out_idx) <= to_std_logic_vector(max_val);
                    max_indices(out_idx) <= max_idx;
                end loop;
            end loop;
        end loop;
    end process;

    -- Backward pass: route gradients to max positions
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                bwd_error_out <= (others => (others => '0'));
            elsif bwd_ctrl_in.valid = '1' then
                -- Zero all input gradients first
                bwd_error_out <= (others => (others => '0'));
                -- Route each output gradient to the position of the max value
                for i in 0 to OUTPUT_SIZE - 1 loop
                    bwd_error_out(max_indices(i)) <= bwd_error_in(i);
                end loop;
            end if;
        end if;
    end process;

end architecture rtl;

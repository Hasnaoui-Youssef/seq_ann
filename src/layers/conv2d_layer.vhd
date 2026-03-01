library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;
use work.layer_interface_pkg.all;

-- Conv2D Layer: 2D convolution with weight sharing
--
-- Input:  C_IN channels of H_IN x W_IN feature maps (flattened to 1D array)
-- Output: NUM_FILTERS channels of H_OUT x W_OUT feature maps
-- Weights stored in weight_bank: (K_H * K_W * C_IN + 1) * NUM_FILTERS values
--
-- Processing: Sequential over output positions, parallel across filters.
-- Each clock cycle computes one output spatial position for all filters.
-- Weight sharing: same kernel weights used across all spatial positions.

entity conv2d_layer is
    generic (
        -- Input feature map dimensions
        C_IN     : integer := 1;
        H_IN     : integer := 28;
        W_IN     : integer := 28;
        -- Convolution parameters
        NUM_FILTERS : integer := 8;
        KERNEL_H    : integer := 3;
        KERNEL_W    : integer := 3;
        STRIDE_H    : integer := 1;
        STRIDE_W    : integer := 1;
        PAD_H       : integer := 0;
        PAD_W       : integer := 0
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Forward Interface (flattened feature maps)
        fwd_ctrl_in  : in layer_control_t;
        fwd_data_in  : in sfixed_bus_array(0 to C_IN * H_IN * W_IN - 1);
        fwd_ctrl_out : out layer_control_t;
        fwd_data_out : out sfixed_bus_array(0 to NUM_FILTERS * ((H_IN + 2*PAD_H - KERNEL_H)/STRIDE_H + 1) * ((W_IN + 2*PAD_W - KERNEL_W)/STRIDE_W + 1) - 1);

        -- Backward Interface
        bwd_ctrl_in  : in layer_control_t;
        bwd_error_in : in sfixed_bus_array(0 to NUM_FILTERS * ((H_IN + 2*PAD_H - KERNEL_H)/STRIDE_H + 1) * ((W_IN + 2*PAD_W - KERNEL_W)/STRIDE_W + 1) - 1);
        bwd_ctrl_out : out layer_control_t;
        bwd_error_out: out sfixed_bus_array(0 to C_IN * H_IN * W_IN - 1);

        -- Weight Bank Interface
        weight_load_en   : in  std_logic;
        weight_load_data : in  sfixed_bus;
        weight_load_done : out std_logic;
        weight_save_en   : in  std_logic;
        weight_save_data : out sfixed_bus;
        weight_save_done : out std_logic;

        -- Gradient update
        weight_update_en   : in  std_logic;
        weight_learn_rate  : in  sfixed_bus;
        weight_update_done : out std_logic;

        -- Gradients output
        grads_out : out sfixed_bus_array(0 to (KERNEL_H * KERNEL_W * C_IN + 1) * NUM_FILTERS - 1)
    );
end entity conv2d_layer;

architecture rtl of conv2d_layer is

    -- Output dimensions
    constant H_OUT : integer := (H_IN + 2*PAD_H - KERNEL_H) / STRIDE_H + 1;
    constant W_OUT : integer := (W_IN + 2*PAD_W - KERNEL_W) / STRIDE_W + 1;
    constant INPUT_SIZE  : integer := C_IN * H_IN * W_IN;
    constant OUTPUT_SIZE : integer := NUM_FILTERS * H_OUT * W_OUT;

    -- Weight layout: per filter: K_H * K_W * C_IN weights + 1 bias
    constant KERNEL_SIZE   : integer := KERNEL_H * KERNEL_W * C_IN;
    constant WEIGHTS_PER_FILTER : integer := KERNEL_SIZE + 1;
    constant NUM_WEIGHTS : integer := WEIGHTS_PER_FILTER * NUM_FILTERS;

    -- Weight storage
    signal weights_internal : sfixed_bus_array(0 to NUM_WEIGHTS - 1);
    signal grads_internal   : sfixed_bus_array(0 to NUM_WEIGHTS - 1)
        := (others => (others => '0'));

    -- Stored inputs for backward pass
    signal stored_input : sfixed_bus_array(0 to INPUT_SIZE - 1)
        := (others => (others => '0'));

    -- Output registers
    signal output_reg : sfixed_bus_array(0 to OUTPUT_SIZE - 1)
        := (others => (others => '0'));

    -- Get padded input value (returns 0 for out-of-bounds)
    function get_padded_input(
        data : sfixed_bus_array;
        c, h, w : integer
    ) return sfixed_bus is
        variable actual_h : integer;
        variable actual_w : integer;
        constant ZERO : sfixed_bus := (others => '0');
    begin
        actual_h := h - PAD_H;
        actual_w := w - PAD_W;
        if actual_h < 0 or actual_h >= H_IN or actual_w < 0 or actual_w >= W_IN then
            return ZERO;
        else
            return data(c * H_IN * W_IN + actual_h * W_IN + actual_w);
        end if;
    end function;

begin

    fwd_data_out <= output_reg;
    grads_out <= grads_internal;

    -- Weight bank instance
    u_weight_bank: entity work.weight_bank
        generic map (NUM_WEIGHTS => NUM_WEIGHTS)
        port map (
            clk => clk, rst => rst,
            load_en => weight_load_en, load_data => weight_load_data,
            load_done => weight_load_done, load_idx => open,
            weights_o => weights_internal,
            update_en => weight_update_en, grad_data => grads_internal,
            learn_rate => weight_learn_rate, update_done => weight_update_done,
            save_en => weight_save_en, save_data => weight_save_data,
            save_done => weight_save_done, save_idx => open
        );

    -- Control signal propagation
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

    -- Forward pass: convolution (combinational)
    process(stored_input, weights_internal)
        variable acc     : sfixed_bus;
        variable prod    : sfixed_bus;
        variable w_base  : integer;
        variable in_h, in_w : integer;
        variable in_val  : sfixed_bus;
        variable w_val   : sfixed_bus;
    begin
        for f in 0 to NUM_FILTERS - 1 loop
            w_base := f * WEIGHTS_PER_FILTER;
            for oh in 0 to H_OUT - 1 loop
                for ow in 0 to W_OUT - 1 loop
                    -- Start with bias
                    acc := weights_internal(w_base + KERNEL_SIZE);

                    -- Accumulate kernel dot product
                    for c in 0 to C_IN - 1 loop
                        for kh in 0 to KERNEL_H - 1 loop
                            for kw in 0 to KERNEL_W - 1 loop
                                in_h := oh * STRIDE_H + kh;
                                in_w := ow * STRIDE_W + kw;
                                in_val := get_padded_input(stored_input, c, in_h, in_w);
                                w_val := weights_internal(w_base + c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw);
                                prod := resize(in_val * w_val, INT_BITS - 1, -FRAC_BITS);
                                acc := resize(acc + prod, INT_BITS - 1, -FRAC_BITS);
                            end loop;
                        end loop;
                    end loop;

                    output_reg(f * H_OUT * W_OUT + oh * W_OUT + ow) <= acc;
                end loop;
            end loop;
        end loop;
    end process;

    -- Store inputs on forward valid
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                stored_input <= (others => (others => '0'));
            elsif fwd_ctrl_in.valid = '1' then
                stored_input <= fwd_data_in;
            end if;
        end if;
    end process;

    -- Backward pass: compute weight gradients and input gradients
    process(clk)
        variable delta_sf : sfixed_bus;
        variable in_sf    : sfixed_bus;
        variable w_sf     : sfixed_bus;
        variable grad_acc : sfixed_bus;
        variable w_base   : integer;
        variable in_h, in_w : integer;
        variable grad_w_acc : sfixed_bus;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                grads_internal <= (others => (others => '0'));
                bwd_error_out <= (others => (others => '0'));

            elsif bwd_ctrl_in.valid = '1' then
                -- Compute weight gradients: dL/dW[f,c,kh,kw] = sum over positions of delta * input
                for f in 0 to NUM_FILTERS - 1 loop
                    w_base := f * WEIGHTS_PER_FILTER;

                    for c in 0 to C_IN - 1 loop
                        for kh in 0 to KERNEL_H - 1 loop
                            for kw in 0 to KERNEL_W - 1 loop
                                grad_w_acc := (others => '0');
                                for oh in 0 to H_OUT - 1 loop
                                    for ow in 0 to W_OUT - 1 loop
                                        delta_sf := bwd_error_in(f * H_OUT * W_OUT + oh * W_OUT + ow);
                                        in_h := oh * STRIDE_H + kh;
                                        in_w := ow * STRIDE_W + kw;
                                        in_sf := get_padded_input(stored_input, c, in_h, in_w);
                                        grad_w_acc := resize(grad_w_acc + resize(delta_sf * in_sf, INT_BITS - 1, -FRAC_BITS),
                                                            INT_BITS - 1, -FRAC_BITS);
                                    end loop;
                                end loop;
                                grads_internal(w_base + c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw)
                                    <= grad_w_acc;
                            end loop;
                        end loop;
                    end loop;

                    -- Bias gradient: sum of deltas
                    grad_w_acc := (others => '0');
                    for oh in 0 to H_OUT - 1 loop
                        for ow in 0 to W_OUT - 1 loop
                            delta_sf := bwd_error_in(f * H_OUT * W_OUT + oh * W_OUT + ow);
                            grad_w_acc := resize(grad_w_acc + delta_sf, INT_BITS - 1, -FRAC_BITS);
                        end loop;
                    end loop;
                    grads_internal(w_base + KERNEL_SIZE) <= grad_w_acc;
                end loop;

                -- Compute input gradients: dL/dX[c,h,w] = sum over filters,kh,kw of delta * weight
                for c in 0 to C_IN - 1 loop
                    for ih in 0 to H_IN - 1 loop
                        for iw in 0 to W_IN - 1 loop
                            grad_acc := (others => '0');
                            for f in 0 to NUM_FILTERS - 1 loop
                                w_base := f * WEIGHTS_PER_FILTER;
                                for kh in 0 to KERNEL_H - 1 loop
                                    for kw in 0 to KERNEL_W - 1 loop
                                        -- Check if this input position contributes to output position (oh, ow)
                                        in_h := ih + PAD_H - kh;
                                        in_w := iw + PAD_W - kw;
                                        if in_h >= 0 and in_h < H_OUT * STRIDE_H and
                                           in_w >= 0 and in_w < W_OUT * STRIDE_W and
                                           in_h mod STRIDE_H = 0 and in_w mod STRIDE_W = 0 then
                                            delta_sf := bwd_error_in(f * H_OUT * W_OUT + (in_h/STRIDE_H) * W_OUT + in_w/STRIDE_W);
                                            w_sf := weights_internal(w_base + c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw);
                                            grad_acc := resize(grad_acc + resize(delta_sf * w_sf, INT_BITS - 1, -FRAC_BITS),
                                                              INT_BITS - 1, -FRAC_BITS);
                                        end if;
                                    end loop;
                                end loop;
                            end loop;
                            bwd_error_out(c * H_IN * W_IN + ih * W_IN + iw) <= grad_acc;
                        end loop;
                    end loop;
                end loop;
            end if;
        end if;
    end process;

end architecture rtl;

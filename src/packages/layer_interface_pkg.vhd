library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;

package layer_interface_pkg is

    -- Layer type enumeration
    type layer_type_t is (LAYER_DENSE, LAYER_CONV2D, LAYER_MAXPOOL, LAYER_AVGPOOL, LAYER_FLATTEN);
    type layer_type_array is array (natural range <>) of layer_type_t;

    -- Feature map dimensions (used for Conv/Pool layers)
    type feature_map_dims_t is record
        channels : integer;  -- Number of channels (C)
        height   : integer;  -- Spatial height (H)
        width    : integer;  -- Spatial width (W)
    end record;

    -- Convolution parameters
    type conv_params_t is record
        kernel_h : integer;
        kernel_w : integer;
        stride_h : integer;
        stride_w : integer;
        pad_h    : integer;
        pad_w    : integer;
    end record;

    -- Pooling parameters
    type pool_params_t is record
        kernel_h : integer;
        kernel_w : integer;
        stride_h : integer;
        stride_w : integer;
    end record;

    -- Index computation helpers for row-major (C, H, W) layout
    function fmap_idx(c, h, w : integer; dims : feature_map_dims_t) return integer;
    function fmap_size(dims : feature_map_dims_t) return integer;

    -- Compute Conv2D output dimensions
    function conv_out_dims(
        input_dims  : feature_map_dims_t;
        num_filters : integer;
        params      : conv_params_t
    ) return feature_map_dims_t;

    -- Compute Pool output dimensions
    function pool_out_dims(
        input_dims : feature_map_dims_t;
        params     : pool_params_t
    ) return feature_map_dims_t;

    -- Compute number of weights for Conv2D: (K_H * K_W * C_in + 1) * C_out
    function conv_weight_count(
        in_channels : integer;
        out_channels : integer;
        params : conv_params_t
    ) return integer;

end package layer_interface_pkg;

package body layer_interface_pkg is

    function fmap_idx(c, h, w : integer; dims : feature_map_dims_t) return integer is
    begin
        return c * dims.height * dims.width + h * dims.width + w;
    end function;

    function fmap_size(dims : feature_map_dims_t) return integer is
    begin
        return dims.channels * dims.height * dims.width;
    end function;

    function conv_out_dims(
        input_dims  : feature_map_dims_t;
        num_filters : integer;
        params      : conv_params_t
    ) return feature_map_dims_t is
        variable result : feature_map_dims_t;
    begin
        result.channels := num_filters;
        result.height := (input_dims.height + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
        result.width  := (input_dims.width  + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;
        return result;
    end function;

    function pool_out_dims(
        input_dims : feature_map_dims_t;
        params     : pool_params_t
    ) return feature_map_dims_t is
        variable result : feature_map_dims_t;
    begin
        result.channels := input_dims.channels;
        result.height := (input_dims.height - params.kernel_h) / params.stride_h + 1;
        result.width  := (input_dims.width  - params.kernel_w) / params.stride_w + 1;
        return result;
    end function;

    function conv_weight_count(
        in_channels : integer;
        out_channels : integer;
        params : conv_params_t
    ) return integer is
    begin
        -- Per filter: K_H * K_W * C_in weights + 1 bias
        return (params.kernel_h * params.kernel_w * in_channels + 1) * out_channels;
    end function;

end package body layer_interface_pkg;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.fixed_float_types.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func is
    generic(
        input_int_bits : integer := 16  -- Integer bits of input (may differ from INT_BITS due to accumulator growth)
    );
    port(
        input_i : in sfixed(input_int_bits - 1 downto -FRAC_BITS);
        output_o : out sfixed_bus  -- Output is always DATA_WIDTH (INT_BITS and FRAC_BITS)
    );
end entity activation_func;

architecture sigmoid of activation_func is

    -- Index calculation signals
    signal index_slice : sfixed(INDEX_HIGH downto INDEX_LOW);
    signal msb_flipped : std_logic;
    signal lut_index : unsigned(LUT_BITS - 1 downto 0);

begin

    ---------------------------------------------------------------------------
    -- Index Calculation via Resize + MSB Flip
    ---------------------------------------------------------------------------
    -- The indexing scheme:
    -- 1. Resize input to index slice range with saturation (clips to [-2^n, 2^n))
    -- 2. Flip MSB to transform signed range to unsigned [0, 2^(n+1))
    -- 3. Use result directly as LUT index
    --
    -- Saturation handles overflow automatically - values outside range clamp
    -- to bounds, which map to LUT[0]=0 and LUT[max]=1 (forced in LUT generation)
    --
    -- Why truncate (floor) instead of round for fractional bits?
    -- 1. Hardware efficiency: truncation is just wire routing, no adder needed
    -- 2. Consistency: always maps to the lower LUT entry of the interval
    -- 3. Bounded error: max error is one step size (2^(n+1-k))
    -- 4. For sigmoid's smooth curve, the difference is negligible
    ---------------------------------------------------------------------------
    index_slice <= resize(
        arg            => input_i,
        left_index     => INDEX_HIGH,
        right_index    => INDEX_LOW,
        overflow_style => fixed_saturate,
        round_style    => fixed_truncate
    );

    -- Flip MSB to transform signed range to unsigned
    -- Example with n=3 (range [-8, 8), 4 integer bits):
    --   -8 = 1000b -> flip MSB -> 0000b = 0  (maps to LUT[0] = 0)
    --   -1 = 1111b -> flip MSB -> 0111b = 7  (maps to middle-ish)
    --    0 = 0000b -> flip MSB -> 1000b = 8  (maps to middle, sigmoid(0)=0.5)
    --   +7 = 0111b -> flip MSB -> 1111b = 15 (maps to LUT[max] = 1)
    msb_flipped <= not index_slice(INDEX_HIGH);

    -- Construct unsigned index: flipped MSB concatenated with remaining bits
    lut_index <= unsigned(msb_flipped & to_slv(index_slice(INDEX_HIGH - 1 downto INDEX_LOW)));

    -- Direct LUT lookup
    output_o <= SIGMOID_LUT(to_integer(lut_index));

end architecture sigmoid;

architecture relu of activation_func is
    constant ZERO : sfixed_bus := to_sfixed(0.0, INT_BITS - 1, -FRAC_BITS);
begin
    -- ReLU: max(0, x) with saturating resize from accumulator width to data width
    output_o <= ZERO when input_i < 0 else
                resize(input_i, INT_BITS - 1, -FRAC_BITS,
                       fixed_saturate, fixed_truncate);
end architecture relu;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.fixed_float_types.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func is
    generic(
        input_width : integer := 48  -- Width of accumulator output (varies by number of inputs)
    );
    port(
        input_i : in std_logic_vector(input_width - 1 downto 0);
        output_o : out sfixed_bus  -- Output is always DATA_WIDTH (INT_BITS and FRAC_BITS)
    );
end entity activation_func;

-- architecture relu of activation_func is
-- begin
--     output_o <= to_sfixed(input_i, output_o) when signed(input_i) >= 0 else to_sfixed_a(0);
-- end architecture relu;

architecture sigmoid of activation_func is
    -- Input integer bits (derived from input width and frac width)
    constant INPUT_INT_BITS : integer := input_width - FRAC_BITS;

    -- Input as sfixed
    signal input_sfixed : sfixed(INPUT_INT_BITS - 1 downto -FRAC_BITS);

    -- Index calculation signals
    -- After MSB flip, extract bits for LUT index
    signal index_slice : sfixed(INDEX_HIGH downto INDEX_LOW);
    signal msb_flipped : std_logic;
    signal lut_index : unsigned(LUT_BITS - 1 downto 0);

begin
    -- Convert input to sfixed
    input_sfixed <= to_sfixed(input_i, input_sfixed);

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
        arg            => input_sfixed,
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

-- architecture clamped_relu of activation_func is
--     constant MAX_THRESHOLD : sfixed((output_width + 1) / 2 - 1 downto -(output_width / 2)) :=
--         to_sfixed(max_value, (output_width + 1) / 2 - 1, -(output_width / 2));
--     constant ZERO : sfixed((output_width + 1) / 2 - 1 downto -(output_width / 2)) :=
--         to_sfixed(0.0, (output_width + 1) / 2 - 1, -(output_width / 2));
--
--     signal input_sfixed : sfixed(input_width - FRAC_BITS - 1 downto -FRAC_BITS);
-- begin
--     -- Startup check
--     assert false report "Clamped ReLU Architecture Instantiated" severity note;
--
--     -- Convert input to sfixed
--     input_sfixed <= to_sfixed(input_i, input_sfixed);
--
--     -- Clamped ReLU: output = clamp(input, 0, max_value)
--     process(input_sfixed)
--     begin
--         if input_sfixed < ZERO then
--             -- Below minimum: clamp to 0
--             output_o <= ZERO;
--         elsif input_sfixed > MAX_THRESHOLD then
--             -- Above maximum: clamp to max_value
--             output_o <= MAX_THRESHOLD;
--         else
--             -- Within range: pass through (resize to output width)
--             output_o <= resize(input_sfixed, output_o);
--         end if;
--     end process;
--
-- end architecture clamped_relu;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;
use work.sigmoid_lut_pkg.all;

entity activation_func is
    generic(
        input_width : integer := 48;  -- Width of accumulator output (varies by number of inputs)
        input_frac_width : integer := FRAC_BITS;  -- Fractional bits (same as DATA_WIDTH format)
        max_value : real := 1.0  -- Maximum threshold for clamped ReLU
    );
    port(
        input_i : in std_logic_vector(input_width - 1 downto 0);
        output_o : out sfixed(INT_BITS - 1 downto -FRAC_BITS) := (others => '0')  -- Output is always DATA_WIDTH
    );
end entity activation_func;

-- architecture relu of activation_func is
-- begin
--     output_o <= to_sfixed(input_i, output_o) when signed(input_i) >= 0 else to_sfixed_a(0);
-- end architecture relu;

architecture sigmoid of activation_func is
    -- Constants for slicing
    constant SLICE_HIGH : integer := 5; -- Sufficient for range -32 to 32 (covers -8 to 8 and prevents wrap-around for 16.0)
    
    -- Signals
    signal input_sfixed : sfixed(input_width - input_frac_width - 1 downto - input_frac_width);
    signal input_slice : sfixed(SLICE_HIGH downto - input_frac_width);
    signal lut_index : integer range 0 to LUT_SIZE - 1;
    signal input_real : real;
    signal clipped : real;
    signal normalized : real;
    
    -- Overflow detection
    signal overflow_pos : boolean;
    signal overflow_neg : boolean;

begin
    -- Convert input to sfixed
    input_sfixed <= to_sfixed(input_i, input_sfixed);

    -- Slicing and Overflow Logic
    process(all) -- Use VHDL-2008 all
        variable v_upper_bits_or : std_logic;
    begin
        -- Default assignments
        overflow_pos <= false;
        overflow_neg <= false;
        input_slice <= (others => '0');
        
        -- Check if input is wider than slice
        if input_sfixed'high > SLICE_HIGH then
            -- Check upper bits for overflow
            v_upper_bits_or := '0';
            for k in input_sfixed'high downto SLICE_HIGH + 1 loop
                if input_sfixed(k) = '1' then
                    v_upper_bits_or := '1';
                end if;
            end loop;
            
            if input_sfixed(input_sfixed'high) = '0' then -- Positive
                -- If any upper bit is 1, it's an overflow
                if v_upper_bits_or = '1' then
                    overflow_pos <= true;
                end if;
            else -- Negative
                -- If any upper bit is 0 (not all ones), it's an overflow (large negative)
                -- Wait, for negative numbers:
                -- -1 is 111...111. Upper bits are 1.
                -- Large negative (e.g. -100) is 11...10...
                -- So we check if upper bits are NOT all ones.
                -- i.e. if any upper bit is 0.
                v_upper_bits_or := '0'; -- Reuse variable to track if any '0' found
                for k in input_sfixed'high downto SLICE_HIGH + 1 loop
                    if input_sfixed(k) = '0' then
                        v_upper_bits_or := '1'; -- Found a zero
                    end if;
                end loop;
                
                if v_upper_bits_or = '1' then
                    overflow_neg <= true;
                end if;
            end if;
            
            input_slice <= input_sfixed(SLICE_HIGH downto -input_frac_width);
        else
            -- Input is smaller than slice
            input_slice <= resize(input_sfixed, input_slice);
        end if;
    end process;

    -- Calculate Real Value with Overflow Handling
    process(all)
    begin
        if overflow_pos then
            input_real <= INPUT_MAX + 1.0; -- Force clip to max
        elsif overflow_neg then
            input_real <= INPUT_MIN - 1.0; -- Force clip to min
        else
            input_real <= to_real(input_slice);
        end if;
        
        report "Sigmoid Debug: slice=" & to_hstring(input_slice) & 
               " real=" & real'image(to_real(input_slice));
    end process;

    clipped <= INPUT_MAX when input_real > INPUT_MAX else INPUT_MIN when input_real < INPUT_MIN else input_real;
    normalized <= ((clipped - INPUT_MIN)/(INPUT_MAX - INPUT_MIN));

    lut_index <= 0 when (normalized  < 0.0)
                 else (LUT_SIZE - 1) when (normalized > 1.0)
                 else integer(normalized * real(LUT_SIZE - 1));


    -- Lookup sigmoid value from LUT (now returns sfixed directly)
    output_o <= SIGMOID_LUT(lut_index);
    
    process(input_real)
    begin
        report "Sigmoid Debug: in=" & real'image(input_real) & 
               " norm=" & real'image(normalized) & 
               " idx=" & integer'image(lut_index) & 
               " val=" & real'image(to_real(SIGMOID_LUT(lut_index)));
    end process;


    
    -- Startup check
    assert false report "Sigmoid Architecture Instantiated" severity note;

end architecture sigmoid;

-- architecture clamped_relu of activation_func is
--     constant MAX_THRESHOLD : sfixed((output_width + 1) / 2 - 1 downto -(output_width / 2)) := 
--         to_sfixed(max_value, (output_width + 1) / 2 - 1, -(output_width / 2));
--     constant ZERO : sfixed((output_width + 1) / 2 - 1 downto -(output_width / 2)) := 
--         to_sfixed(0.0, (output_width + 1) / 2 - 1, -(output_width / 2));
--     
--     signal input_sfixed : sfixed(input_width - input_frac_width - 1 downto -input_frac_width);
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

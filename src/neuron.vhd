library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity neuron is
    generic (
        layer_index : integer := 0;
        neuron_index : integer := 0;
        num_inputs : integer := 4;      -- Number of input values (excluding bias)
        use_sigmoid : boolean := true   -- true for sigmoid, false for relu
    );
    port (
        -- Clock for weight loading
        clk : in std_logic;
        
        -- Inputs: array of input values
        inputs_i : in std_logic_bus_array(0 to num_inputs - 1)(DATA_WIDTH - 1 downto 0);



        -- Weight loading interface
        load_enable : in std_logic;     -- Enable weight loading
        weight_data_i : in std_logic_vector(DATA_WIDTH - 1 downto 0);  -- Weight data
        weight_index_i : in integer;  -- Which weight to load

        -- Output: activated result
        output_o : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Optional overflow indicator from accumulator
        overflow_o : out std_logic
    );
end entity neuron;

architecture rtl of neuron is
    -- Internal weight storage
    signal weights_reg : std_logic_bus_array(0 to num_inputs - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));
    signal bias_reg : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    
    -- Active weights (either from registers or external)
    signal weights_active : std_logic_bus_array(0 to num_inputs)(DATA_WIDTH - 1 downto 0);

    -- Fixed-point array type for multiplication results
    -- Result of multiplying two (INT.FRAC) numbers has (2*INT).(2*FRAC) bits
    -- We truncate the fractional part back to FRAC bits
    -- So result is (2*INT).FRAC
    constant MULT_INT_BITS : integer := 2 * NUM_INT_BITS;
    
    type sfixed_mult_array is array (integer range<>) of sfixed(MULT_INT_BITS - 1 downto -NUM_FRAC_BITS);

    signal mult_results : sfixed_mult_array(num_inputs downto 0);

    -- Accumulator inputs (mult results + bias)
    -- Width is MULT_INT_BITS + NUM_FRAC_BITS
    constant ACC_WIDTH : integer := MULT_INT_BITS + NUM_FRAC_BITS;
    signal acc_inputs : std_logic_bus_array(num_inputs downto 0)(ACC_WIDTH - 1 downto 0);

    -- Accumulator output
    -- Summing num_inputs+1 values adds log2(num_inputs+1) bits
    -- We'll just add num_inputs bits to be safe/simple as per original code style
    signal acc_sum : std_logic_vector(ACC_WIDTH + num_inputs downto 0);
    signal acc_overflow : std_logic;

    signal activation_input : std_logic_vector(ACC_WIDTH + num_inputs downto 0);
    signal activation_output : sfixed(NUM_INT_BITS - 1 downto -NUM_FRAC_BITS);

begin
    -- ========================================================================
    -- Step 0: Weight Loading and Selection
    -- ========================================================================
    
    -- Weight loading process
    weight_load_proc : process(clk)
    begin
        if rising_edge(clk) then
            if load_enable = '1' then
                if weight_index_i < num_inputs then
                    -- Load a weight
                    weights_reg(weight_index_i) <= weight_data_i;
                else
                    -- Load bias
                    bias_reg <= weight_data_i;
                end if;
            end if;
        end if;
    end process;
    
    -- Select active weights (use stored weights)
    weights_active(0 to num_inputs - 1) <= weights_reg;
    weights_active(num_inputs) <= bias_reg;

    -- ========================================================================
    -- Step 1: Multiply inputs by weights
    -- ========================================================================
    mult_gen : for i in 0 to num_inputs - 1 generate
        process(inputs_i, weights_active)
            variable v_input : sfixed(NUM_INT_BITS - 1 downto -NUM_FRAC_BITS);
            variable v_weight : sfixed(NUM_INT_BITS - 1 downto -NUM_FRAC_BITS);
            variable v_mult_full : sfixed(2*NUM_INT_BITS - 1 downto -2*NUM_FRAC_BITS);
        begin
            v_input := to_sfixed(inputs_i(i), NUM_INT_BITS - 1, -NUM_FRAC_BITS);
            v_weight := to_sfixed(weights_active(i), NUM_INT_BITS - 1, -NUM_FRAC_BITS);
            
            v_mult_full := v_input * v_weight;
            
            -- Resize with truncation (keeping upper bits, discarding lower fractional bits)
            -- Target: MULT_INT_BITS - 1 downto -NUM_FRAC_BITS
            mult_results(i) <= resize(v_mult_full, MULT_INT_BITS - 1, -NUM_FRAC_BITS);
        end process;
    end generate mult_gen;

    -- ========================================================================
    -- Step 2: Prepare accumulator inputs (products + bias)
    -- ========================================================================
    -- First num_inputs entries are multiplication results
    acc_input_assign : for i in 0 to num_inputs - 1 generate
        acc_inputs(i) <= to_slv(mult_results(i));
    end generate acc_input_assign;

    -- Last entry is the bias
    bias_extend : process(weights_active)
        variable v_bias : sfixed(NUM_INT_BITS - 1 downto -NUM_FRAC_BITS);
        variable v_bias_extended : sfixed(MULT_INT_BITS - 1 downto -NUM_FRAC_BITS);
    begin
        v_bias := to_sfixed(weights_active(num_inputs), NUM_INT_BITS - 1, -NUM_FRAC_BITS);
        
        -- Resize bias to match multiplication result width (sign extend integer part)
        v_bias_extended := resize(v_bias, MULT_INT_BITS - 1, -NUM_FRAC_BITS);
        
        acc_inputs(num_inputs) <= to_slv(v_bias_extended);
    end process;

    -- ========================================================================
    -- Step 3: Accumulate all values (weights*inputs + bias)
    -- ========================================================================
    -- Accumulator instance
    acc_inst : entity work.acc
        generic map (
            size => num_inputs + 1, -- +1 for bias
            data_width => ACC_WIDTH,
            signed_acc => true
        )
        port map (
            inputs => acc_inputs,
            sum_o => acc_sum,
            overflow_o => acc_overflow
        );

    -- Connect overflow output
    overflow_o <= acc_overflow;

    -- ========================================================================
    -- Step 4: Apply activation function
    -- ========================================================================
    -- Pass the full accumulator sum to activation function
    -- The activation function should handle the resizing/clamping
    activation_input <= acc_sum;
    
    -- Choose activation function based on generic
    sigmoid_gen : if use_sigmoid generate
        activation_inst : entity work.activation_func(sigmoid)
            generic map(
                input_width => ACC_WIDTH + num_inputs + 1,
                input_frac_width => NUM_FRAC_BITS, -- We maintained this fractional width
                output_width => DATA_WIDTH
            )
            port map(
                input_i => activation_input,
                output_o => activation_output
            );
    end generate sigmoid_gen;

    relu_gen : if not use_sigmoid generate
        activation_inst : entity work.activation_func(clamped_relu)
            generic map(
                input_width => ACC_WIDTH + num_inputs + 1,
                input_frac_width => NUM_FRAC_BITS,
                output_width => DATA_WIDTH
            )
            port map(
                input_i => activation_input,
                output_o => activation_output
            );
    end generate relu_gen;

    -- Convert activation output back to std_logic_vector
    output_o <= to_slv(activation_output);



end architecture rtl;

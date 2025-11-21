library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

entity neuron is
    generic (
        num_inputs : integer := 4;      -- Number of input values (excluding bias)
        data_width : integer := 16;     -- Bit width of each input/weight
        use_sigmoid : boolean := true   -- true for sigmoid, false for relu
    );
    port (
        -- Clock for weight loading
        clk : in std_logic;
        
        -- Inputs: array of input values
        inputs_i : in std_logic_bus_array(num_inputs - 1 downto 0)(data_width - 1 downto 0);

        -- External weights (for compatibility/bypass mode)
        -- weights_i(0 to num_inputs-1) are weights for inputs
        -- weights_i(num_inputs) is the bias
        weights_i : in std_logic_bus_array(0 to num_inputs)(data_width - 1 downto 0);

        -- Weight loading interface
        load_enable : in std_logic;     -- Enable weight loading
        weight_data_i : in std_logic_vector(data_width - 1 downto 0);  -- Weight data
        weight_index_i : in integer range 0 to 15;  -- Which weight to load (0 to num_inputs-1 = weights, num_inputs = bias)

        -- Output: activated result
        output_o : out std_logic_vector(data_width - 1 downto 0);

        -- Optional overflow indicator from accumulator
        overflow_o : out std_logic
    );
end entity neuron;

architecture rtl of neuron is
    -- Internal weight storage
    signal weights_reg : std_logic_bus_array(0 to num_inputs - 1)(data_width - 1 downto 0) := (others => (others => '0'));
    signal bias_reg : std_logic_vector(data_width - 1 downto 0) := (others => '0');
    
    -- Active weights (either from registers or external)
    signal weights_active : std_logic_bus_array(0 to num_inputs)(data_width - 1 downto 0);

    -- Fixed-point array type
    type sfixed_vector_array is array (integer range<>) of sfixed((2*data_width + 1) / 2 - 1 downto -(2*data_width / 2));

    signal mult_results : sfixed_vector_array(num_inputs downto 0);

    -- Accumulator inputs (mult results + bias)
    signal acc_inputs : std_logic_bus_array(num_inputs downto 0)((2 * data_width) - 1 downto 0);

    -- Accumulator output
    signal acc_sum : std_logic_vector((2*data_width) + num_inputs downto 0);
    signal acc_overflow : std_logic;

    signal activation_input : std_logic_vector((2*data_width) + num_inputs - (data_width/2) downto 0);
    signal activation_output : sfixed((data_width + 1) / 2 - 1 downto -(data_width / 2));

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
        mult_results(i) <= resize(to_sfixed(inputs_i(i),(data_width + 1) / 2 - 1, -(data_width / 2)) * to_sfixed(weights_active(i), (data_width + 1) / 2 - 1, -(data_width / 2)), mult_results(i)'high, mult_results(i)'low);
    end generate mult_gen;

    -- ========================================================================
    -- Step 2: Prepare accumulator inputs (products + bias)
    -- ========================================================================
    -- First num_inputs entries are multiplication results
    acc_input_assign : for i in 0 to num_inputs - 1 generate
        acc_inputs(i) <= to_slv(mult_results(i));
    end generate acc_input_assign;

    -- Last entry is the bias (sign-extended to match width)
    bias_extend : process(weights_active)
    begin
        -- Sign extend bias to MULT_OUTPUT_WIDTH
        -- Bias is data_width bits (frac=data_width/2)
        -- Mult results are 2*data_width bits (frac=data_width)
        -- We need to shift bias left by data_width/2 to align fractional points
        
        -- Sign extension
        acc_inputs(num_inputs)((2*data_width) - 1 downto data_width + (data_width/2)) <=
            (others => weights_active(num_inputs)(data_width - 1));  -- Sign bit
            
        -- Bias value
        acc_inputs(num_inputs)(data_width + (data_width/2) - 1 downto data_width/2) <=
            weights_active(num_inputs);
            
        -- Zero padding for lower fractional bits
        acc_inputs(num_inputs)((data_width/2) - 1 downto 0) <= (others => '0');
    end process;

    -- ========================================================================
    -- Step 3: Accumulate all values (weights*inputs + bias)
    -- ========================================================================
    accumulator_inst : entity work.acc
        generic map(
            size => num_inputs + 1,
            data_width => 2 * data_width,
            signed_acc => true  -- We're working with signed values
        )
        port map(
            inputs => acc_inputs,
            sum_o => acc_sum,
            overflow_o => acc_overflow
        );

    -- Connect overflow output
    overflow_o <= acc_overflow;

    -- ========================================================================
    -- Step 3.5: Resize accumulator output to activation input width
    -- ========================================================================
    --resize_proc : process(acc_sum)
    --    variable acc_signed : signed(ACC_OUTPUT_WIDTH - 1 downto 0);
    --begin
    --    acc_signed := signed(acc_sum);

    --    if ACC_OUTPUT_WIDTH >= ACTIVATION_INPUT_WIDTH then
    --        -- Truncate from MSB side (keep lower bits which have more precision)
    --        activation_input <= std_logic_vector(acc_signed(ACTIVATION_INPUT_WIDTH - 1 downto 0));
    --    else
    --        -- Sign extend if accumulator output is smaller
    --        activation_input(ACTIVATION_INPUT_WIDTH - 1 downto ACC_OUTPUT_WIDTH) <=
    --            (others => acc_sum(ACC_OUTPUT_WIDTH - 1));  -- Sign bit
    --        activation_input(ACC_OUTPUT_WIDTH - 1 downto 0) <= acc_sum;
    --    end if;
    --end process;

    -- ========================================================================
    -- Step 4: Apply activation function
    -- ========================================================================
    -- ========================================================================
    -- Step 4: Apply activation function
    -- ========================================================================
    -- Truncate fractional bits to match activation function expectation (data_width/2)
    -- Accumulator has data_width fractional bits (due to multiplication)
    -- We drop the lower data_width/2 bits
    activation_input <= acc_sum(acc_sum'high downto data_width/2);

    -- Choose activation function based on generic
    sigmoid_gen : if use_sigmoid generate
        activation_inst : entity work.activation_func(sigmoid)
            generic map(
                input_width => (2*data_width) + num_inputs + 1 - (data_width/2),
                output_width => data_width
            )
            port map(
                input_i => activation_input,
                output_o => activation_output
            );
    end generate sigmoid_gen;

    relu_gen : if not use_sigmoid generate
        activation_inst : entity work.activation_func(relu)
            generic map(
                input_width => (2*data_width) + num_inputs + 1 - (data_width/2),
                output_width => data_width
            )
            port map(
                input_i => activation_input,
                output_o => activation_output
            );
    end generate relu_gen;

    -- Convert activation output back to std_logic_vector
    output_o <= to_slv(activation_output);

end architecture rtl;

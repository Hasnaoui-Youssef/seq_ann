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
        -- Inputs: array of input values
        inputs_i : in std_logic_bus_array(num_inputs - 1 downto 0)(data_width - 1 downto 0);

        -- Weights: array of weights (num_inputs weights + 1 bias)
        -- weights_i(0 to num_inputs-1) are weights for inputs
        -- weights_i(num_inputs) is the bias
        weights_i : in std_logic_bus_array(num_inputs downto 0)(data_width - 1 downto 0);

        -- Output: activated result
        output_o : out std_logic_vector(data_width - 1 downto 0);

        -- Optional overflow indicator from accumulator
        overflow_o : out std_logic
    );
end entity neuron;

architecture rtl of neuron is
    -- Calculate accumulator size
    -- For num_inputs multiplications + bias: need num_inputs + 1 values to accumulate
    constant ACC_SIZE : integer := num_inputs + 1;
    constant MULT_OUTPUT_WIDTH : integer := 2 * data_width;  -- Multiplier output width

    -- According to acc.vhd: sum_o is (size + data_width - 1 downto 0)
    -- where size = ACC_SIZE and data_width parameter in acc = MULT_OUTPUT_WIDTH
    constant ACC_OUTPUT_WIDTH : integer := ACC_SIZE + MULT_OUTPUT_WIDTH;

    -- Multiplication results (input * weight)
    signal mult_results : std_logic_bus_array(num_inputs - 1 downto 0)(MULT_OUTPUT_WIDTH - 1 downto 0);

    -- Accumulator inputs (mult results + bias)
    signal acc_inputs : std_logic_bus_array(ACC_SIZE - 1 downto 0)(MULT_OUTPUT_WIDTH - 1 downto 0);

    -- Accumulator output
    signal acc_sum : std_logic_vector(ACC_OUTPUT_WIDTH - 1 downto 0);
    signal acc_overflow : std_logic;

    -- Activation function signals
    -- We need to resize the accumulator output to match activation function input expectations
    -- The activation function expects input_width bits (default 48 for your config)
    constant ACTIVATION_INPUT_WIDTH : integer := ACC_OUTPUT_WIDTH;  -- Matches your sigmoid LUT config
    signal activation_input : std_logic_vector(ACTIVATION_INPUT_WIDTH - 1 downto 0);
    signal activation_output : sfixed(data_width / 2 - 1 downto -(data_width / 2));

begin
    -- ========================================================================
    -- Step 1: Multiply inputs by weights
    -- ========================================================================
    mult_gen : for i in 0 to num_inputs - 1 generate
        mult_inst : entity work.mult
            generic map(
                data_width => data_width
            )
            port map(
                a_i => inputs_i(i),
                b_i => weights_i(i),
                product_o => mult_results(i)
            );
    end generate mult_gen;

    -- ========================================================================
    -- Step 2: Prepare accumulator inputs (products + bias)
    -- ========================================================================
    -- First num_inputs entries are multiplication results
    acc_input_assign : for i in 0 to num_inputs - 1 generate
        acc_inputs(i) <= mult_results(i);
    end generate acc_input_assign;

    -- Last entry is the bias (sign-extended to match width)
    bias_extend : process(weights_i)
    begin
        -- Sign extend bias to MULT_OUTPUT_WIDTH
        acc_inputs(num_inputs)(MULT_OUTPUT_WIDTH - 1 downto data_width) <=
            (others => weights_i(num_inputs)(data_width - 1));  -- Sign bit
        acc_inputs(num_inputs)(data_width - 1 downto 0) <=
            weights_i(num_inputs);
    end process;

    -- ========================================================================
    -- Step 3: Accumulate all values (weights*inputs + bias)
    -- ========================================================================
    accumulator_inst : entity work.acc
        generic map(
            size => ACC_SIZE,
            data_width => MULT_OUTPUT_WIDTH,
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

    -- Choose activation function based on generic
    sigmoid_gen : if use_sigmoid generate
        activation_inst : entity work.activation_func(sigmoid)
            generic map(
                input_width => ACTIVATION_INPUT_WIDTH,
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
                input_width => ACTIVATION_INPUT_WIDTH,
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

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.sigmoid_lut_pkg.all;

entity neuron is
    generic(
        NUM_INPUTS : integer := 4;
        USE_SIGMOID : boolean := true
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Forward Pass
        fwd_en       : in std_logic;
        inputs_i     : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        weights_i    : in std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        bias_i       : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_o     : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Backward Pass
        bwd_en       : in std_logic;
        error_i      : in std_logic_vector(DATA_WIDTH - 1 downto 0); -- dL/dy
        
        -- Gradients Output
        grad_weights_o : out std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
        grad_bias_o    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        grad_inputs_o  : out std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0) -- dL/dx
    );
end entity neuron;

architecture rtl of neuron is

    -- Internal Storage for Forward Pass (needed for Backward)
    signal stored_inputs : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
    signal stored_output_reg : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal stored_deriv  : std_logic_vector(DATA_WIDTH - 1 downto 0); -- f'(net)

    -- Internal Signals
    signal sum_sig : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
    signal act_out_sig : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);

    -- Helper for fixed-point multiplication
    function mult(a, b : std_logic_vector) return std_logic_vector is
        variable res : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
    begin
        res := resize(to_sfixed(a, DATA_WIDTH/2 - 1, -DATA_WIDTH/2) * 
                      to_sfixed(b, DATA_WIDTH/2 - 1, -DATA_WIDTH/2), 
                      DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
        return to_std_logic_vector(res);
    end function;

    -- Helper for sigmoid derivative: y * (1 - y)
    function calc_sigmoid_deriv(y : std_logic_vector) return std_logic_vector is
        variable y_sf : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable one_sf : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable res : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
    begin
        y_sf := to_sfixed(y, DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
        one_sf := to_sfixed(1.0, DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
        res := resize(y_sf * (one_sf - y_sf), DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
        return to_std_logic_vector(res);
    end function;

begin

    -- Output is registered value
    output_o <= stored_output_reg;

    -- Instantiate Activation Function
    u_act: entity work.activation_func(sigmoid)
        generic map (
            input_width => DATA_WIDTH,
            input_frac_width => DATA_WIDTH/2,
            output_width => DATA_WIDTH
        )
        port map (
            input_i => to_std_logic_vector(sum_sig),
            output_o => act_out_sig
        );

    process(clk, rst)
        variable delta : std_logic_vector(DATA_WIDTH - 1 downto 0); -- dL/dy * f'(net)
    begin
        if rst = '1' then
            stored_output_reg <= (others => '0');
            stored_deriv <= (others => '0');
            grad_bias_o <= (others => '0');
            -- sum_sig <= (others => '0'); -- Removed to avoid multiple drivers
            
        elsif rising_edge(clk) then
            
            -- Forward Pass
            if fwd_en = '1' then
                -- Store inputs for backward pass
                stored_inputs <= inputs_i;
                
                -- Latch Output
                if USE_SIGMOID then
                    report "Neuron Latch: en=" & std_logic'image(fwd_en) & " act_out=" & to_hstring(act_out_sig) & " stored=" & to_hstring(stored_output_reg);
                    stored_output_reg <= to_std_logic_vector(act_out_sig);
                else
                    stored_output_reg <= to_std_logic_vector(sum_sig);
                end if;
                
                -- Derivative: y * (1 - y)
                -- stored_deriv <= calc_sigmoid_deriv(stored_output_reg);
                -- For now, set deriv to 1.0 (simplified)
                stored_deriv <= to_std_logic_vector(to_sfixed(1.0, DATA_WIDTH/2 - 1, -DATA_WIDTH/2));
            end if;

            -- Backward Pass
            if bwd_en = '1' then
                -- Calculate Delta = Error * Derivative
                delta := mult(error_i, stored_deriv);
                
                -- Gradient for Bias: dL/db = delta
                grad_bias_o <= delta;
                
                -- Gradients for Weights: dL/dw = delta * input
                -- Gradients for Inputs: dL/dx = delta * weight
                for i in 0 to NUM_INPUTS - 1 loop
                    grad_weights_o(i) <= mult(delta, stored_inputs(i));
                    grad_inputs_o(i) <= mult(delta, weights_i(i));
                end loop;
            end if;
            
        end if;
    end process;
    
    -- Combinatorial Sum Calculation
    process(fwd_en, inputs_i, weights_i, bias_i)
        variable sum : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable term : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable v_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
        variable v_w : std_logic_vector(DATA_WIDTH - 1 downto 0);
    begin
        if fwd_en = '1' then
            sum := to_sfixed(bias_i, DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
            for i in 0 to NUM_INPUTS - 1 loop
                v_in := inputs_i(i);
                v_w := weights_i(i);
                term := resize(to_sfixed(v_in, DATA_WIDTH/2 - 1, -DATA_WIDTH/2) * 
                               to_sfixed(v_w, DATA_WIDTH/2 - 1, -DATA_WIDTH/2), 
                               DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
                sum := resize(sum + term, DATA_WIDTH/2 - 1, -DATA_WIDTH/2);
                
                report "Neuron Step: i=" & integer'image(i) & 
                       " in=" & to_hstring(v_in) & 
                       " w=" & to_hstring(v_w) & 
                       " term=" & to_string(to_real(term));
            end loop;
            sum_sig <= sum;
            report "Neuron Result: sum=" & to_string(to_real(sum)) & 
                   " out=" & to_hstring(stored_output_reg);
        else
            sum_sig <= (others => '0');
        end if;
    end process;

end architecture rtl;

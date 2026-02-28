library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity xor_tb is
end entity xor_tb;

architecture testbench of xor_tb is
    -- Configuration
    -- NOTE: The clock period is set to 100 ns (10x slower than typical 10 ns)
    -- to accommodate longer combinational paths through multi-layer network
    -- and improve simulation visibility for debugging purposes.
    constant CLK_PERIOD : time := 100 ns;
    constant TOLERANCE_C : real := 0.1;

    -- Network topology: 2 inputs -> 3 hidden -> 1 output
    constant NUM_INPUTS : integer := 2;
    constant NUM_LAYERS : integer := 2;
    constant LAYER_SIZES : layer_config_array(0 to 1) := (3, 1);
    constant ACTIVATIONS : boolean_array(0 to 1) := (true, true);  -- sigmoid for all layers

    -- Signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal load_weights : std_logic := '0';
    signal start : std_logic := '0';
    signal train_mode : std_logic := '0';
    signal weights_loaded : std_logic;
    signal ready : std_logic;
    signal done : std_logic;

    -- Host Interface
    signal host_write_en : std_logic := '0';
    signal host_addr : integer := 0;
    signal host_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');

    -- Output Interface
    signal output_data : std_logic_bus_array(0 to 0)(DATA_WIDTH - 1 downto 0);
    signal output_valid : std_logic;

    -- Helper to convert real to std_logic_vector
    function to_slv(r : real) return std_logic_vector is
    begin
        return to_std_logic_vector(to_sfixed(r, INT_BITS - 1, -FRAC_BITS));
    end function;

    -- Helper to convert std_logic_vector to real
    function to_real_val(slv : std_logic_vector) return real is
    begin
        return to_real(to_sfixed(slv, INT_BITS - 1, -FRAC_BITS));
    end function;

begin

    -- Clock Generation
    process
    begin
        while true loop
            clk <= '0'; wait for CLK_PERIOD / 2;
            clk <= '1'; wait for CLK_PERIOD / 2;
        end loop;
    end process;

    -- DUT Instantiation
    dut: entity work.neural_network
        generic map (
            NUM_INPUTS  => NUM_INPUTS,
            NUM_LAYERS  => NUM_LAYERS,
            LAYER_SIZES => LAYER_SIZES,
            USE_SIGMOID => ACTIVATIONS
        )
        port map (
            clk => clk,
            rst => rst,
            load_weights => load_weights,
            start => start,
            train_mode => train_mode,
            weights_loaded => weights_loaded,
            ready => ready,
            done => done,
            host_write_en => host_write_en,
            host_addr => host_addr,
            host_data => host_data,
            output_data => output_data,
            output_valid => output_valid
        );

    -- Test Process
    process
        -- Memory layout:
        -- Addr 0: input[0]
        -- Addr 1: input[1]
        -- Addr 2+: weights

        procedure write_mem(addr : integer; val : real) is
        begin
            host_addr <= addr;
            host_data <= to_slv(val);
            host_write_en <= '1';
            wait until rising_edge(clk);
            host_write_en <= '0';
            wait until rising_edge(clk);
        end procedure;

        procedure run_inference(in1 : real; in2 : real; expected : real) is
        begin
            -- Make sure previous inference is fully complete
            while done = '1' loop
                wait until rising_edge(clk);
            end loop;

            -- Load inputs to memory
            write_mem(0, in1);
            write_mem(1, in2);

            -- Ensure system is ready
            while ready /= '1' loop
                wait until rising_edge(clk);
            end loop;

            -- Start inference
            start <= '1';
            train_mode <= '0';
            wait until rising_edge(clk);
            start <= '0';

            -- Wait for done
            wait until done = '1';

            report "Input: " & real'image(in1) & ", " & real'image(in2) &
                   " | Output: " & real'image(to_real_val(output_data(0))) &
                   " | Expected: " & real'image(expected);

            assert abs(to_real_val(output_data(0)) - expected) < TOLERANCE_C
                report "Test Failed!" severity error;

        end procedure;

    begin
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD * 2;

        report "Loading Weights...";
        -- Memory layout: inputs at 0-1, weights at 2+
        -- Layer 0 (3 neurons, 2 inputs + bias each) = 9 weights at addr 2-10
        -- N0 (OR-like): w=[10, 10], b=-5
        write_mem(2, 10.0); write_mem(3, 10.0); write_mem(4, -5.0);
        -- N1 (NAND-like): w=[-10, -10], b=15
        write_mem(5, -10.0); write_mem(6, -10.0); write_mem(7, 15.0);
        -- N2 (Unused/Zero): w=[0, 0], b=0
        write_mem(8, 0.0); write_mem(9, 0.0); write_mem(10, 0.0);

        -- Layer 1 (1 neuron, 3 inputs + bias) = 4 weights at addr 11-14
        -- N0 (AND-like): w=[10, 10, 0], b=-15
        write_mem(11, 10.0); write_mem(12, 10.0); write_mem(13, 0.0); write_mem(14, -15.0);

        report "Weights Loaded. Triggering Weight Load...";

        -- Trigger weight loading from memory to weight banks
        load_weights <= '1';
        wait until rising_edge(clk);
        load_weights <= '0';

        -- Wait for weights to be loaded
        wait until weights_loaded = '1';
        report "Weight Banks Ready. Starting Inference Tests...";

        run_inference(0.0, 0.0, 0.0); -- 0 XOR 0 = 0
        run_inference(0.0, 1.0, 1.0); -- 0 XOR 1 = 1
        run_inference(1.0, 0.0, 1.0); -- 1 XOR 0 = 1
        run_inference(1.0, 1.0, 0.0); -- 1 XOR 1 = 0

        report "All XOR Tests Complete!";
        wait for 2 us;
        std.env.stop;
        wait;
    end process;

end architecture testbench;

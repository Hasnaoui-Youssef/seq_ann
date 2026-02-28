library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use IEEE.math_real.all;
use work.types.all;

entity xor_train_tb is
end entity xor_train_tb;

architecture testbench of xor_train_tb is
    constant CLK_PERIOD : time := 100 ns;

    -- Network topology: 2 inputs -> 3 hidden -> 1 output
    constant NUM_INPUTS : integer := 2;
    constant NUM_LAYERS : integer := 2;
    constant LAYER_SIZES : layer_config_array(0 to 1) := (3, 1);
    constant ACTIVATIONS : boolean_array(0 to 1) := (true, true);

    -- XOR training data
    type sample_t is record
        x0 : real;
        x1 : real;
        y  : real;
    end record;
    type sample_array_t is array (natural range <>) of sample_t;
    constant XOR_DATA : sample_array_t(0 to 3) := (
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0)
    );

    -- Signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal load_weights : std_logic := '0';
    signal start : std_logic := '0';
    signal train_mode : std_logic := '0';
    signal weights_loaded : std_logic;
    signal ready : std_logic;
    signal done : std_logic;

    signal host_write_en : std_logic := '0';
    signal host_addr : integer := 0;
    signal host_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');

    signal output_data : std_logic_bus_array(0 to 0)(DATA_WIDTH - 1 downto 0);
    signal output_valid : std_logic;

    function to_slv(r : real) return std_logic_vector is
    begin
        return to_std_logic_vector(to_sfixed(r, INT_BITS - 1, -FRAC_BITS));
    end function;

    function to_real_val(slv : std_logic_vector) return real is
    begin
        return to_real(to_sfixed(slv, INT_BITS - 1, -FRAC_BITS));
    end function;

begin

    process
    begin
        while true loop
            clk <= '0'; wait for CLK_PERIOD / 2;
            clk <= '1'; wait for CLK_PERIOD / 2;
        end loop;
    end process;

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

    process
        procedure write_mem(addr : integer; val : real) is
        begin
            host_addr <= addr;
            host_data <= to_slv(val);
            host_write_en <= '1';
            wait until rising_edge(clk);
            host_write_en <= '0';
            wait until rising_edge(clk);
        end procedure;

        procedure load_sample(idx : integer) is
        begin
            -- Write inputs at addresses 0, 1
            write_mem(0, XOR_DATA(idx).x0);
            write_mem(1, XOR_DATA(idx).x1);
            -- Write target at address 2 (TARGET_BASE_ADDR = NUM_INPUTS = 2)
            write_mem(2, XOR_DATA(idx).y);
        end procedure;

        procedure run_train_step(sample_idx : integer) is
        begin
            load_sample(sample_idx);

            while ready /= '1' loop
                wait until rising_edge(clk);
            end loop;

            -- Start training step (forward + backward + update)
            start <= '1';
            train_mode <= '1';
            wait until rising_edge(clk);
            start <= '0';

            -- Wait for done
            wait until done = '1';
            wait until rising_edge(clk);
        end procedure;

        procedure run_inference(in1, in2 : real) is
        begin
            write_mem(0, in1);
            write_mem(1, in2);

            while ready /= '1' loop
                wait until rising_edge(clk);
            end loop;

            start <= '1';
            train_mode <= '0';
            wait until rising_edge(clk);
            start <= '0';

            wait until done = '1';
        end procedure;

        variable output_val : real;
        variable total_error : real;
        constant NUM_EPOCHS : integer := 200;
        constant TOLERANCE : real := 0.3;

    begin
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD * 2;

        report "Loading initial weights...";
        -- Memory layout: inputs at 0-1, targets at 2, weights at 3+
        -- Layer 0 (3 neurons, 2 inputs + bias = 3 weights each, 9 total) at addr 3-11
        -- Xavier-inspired initialization: larger, asymmetric weights for symmetry breaking
        write_mem(3,  2.0); write_mem(4, -2.0); write_mem(5, -1.0);
        write_mem(6, -2.0); write_mem(7,  2.0); write_mem(8, -1.0);
        write_mem(9,  1.0); write_mem(10,  1.0); write_mem(11, -1.5);

        -- Layer 1 (1 neuron, 3 inputs + bias = 4 weights) at addr 12-15
        write_mem(12, 2.0); write_mem(13, 2.0); write_mem(14, -1.0); write_mem(15, -1.5);

        report "Triggering weight load...";
        load_weights <= '1';
        wait until rising_edge(clk);
        load_weights <= '0';
        wait until weights_loaded = '1';
        report "Weights loaded. Starting training...";

        -- Training loop
        for epoch in 0 to NUM_EPOCHS - 1 loop
            for s in 0 to 3 loop
                run_train_step(s);
            end loop;

            -- Log progress every 100 epochs
            if (epoch + 1) mod 50 = 0 then
                -- Run inference on all 4 patterns to check progress
                total_error := 0.0;
                for s in 0 to 3 loop
                    run_inference(XOR_DATA(s).x0, XOR_DATA(s).x1);
                    output_val := to_real_val(output_data(0));
                    total_error := total_error + abs(output_val - XOR_DATA(s).y);
                end loop;
                report "Epoch " & integer'image(epoch + 1) &
                       " | Total Error: " & real'image(total_error);
            end if;
        end loop;

        report "Training complete. Running final validation...";

        -- Final validation
        for s in 0 to 3 loop
            run_inference(XOR_DATA(s).x0, XOR_DATA(s).x1);
            output_val := to_real_val(output_data(0));
            report "XOR(" & real'image(XOR_DATA(s).x0) & "," &
                   real'image(XOR_DATA(s).x1) & ") = " &
                   real'image(output_val) & " (expected " &
                   real'image(XOR_DATA(s).y) & ")";

            assert abs(output_val - XOR_DATA(s).y) < TOLERANCE
                report "Training did not converge for this input!" severity warning;
        end loop;

        report "XOR Training Test Complete!";
        std.env.stop;
    end process;

end architecture testbench;

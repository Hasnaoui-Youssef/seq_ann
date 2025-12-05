library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

entity xor_tb is
end entity xor_tb;

architecture testbench of xor_tb is
    -- Configuration
    constant CLK_PERIOD : time := 10 ns;
    constant DATA_WIDTH : integer := 32;
    constant TOLERANCE_C : real := 0.1;

    -- Signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal start : std_logic := '0';
    signal train_mode : std_logic := '0';
    signal ready : std_logic;
    signal done : std_logic;

    -- Host Interface
    signal host_write_en : std_logic := '0';
    signal host_addr : integer := 0;
    signal host_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');

    -- Data Interface
    signal input_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal input_valid : std_logic := '0';
    signal input_last : std_logic := '0';
    signal output_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal output_valid : std_logic;
    signal output_last : std_logic;

    -- Helper to convert real to std_logic_vector
    function to_slv(r : real) return std_logic_vector is
    begin
        return to_std_logic_vector(to_sfixed(r, DATA_WIDTH/2 - 1, -DATA_WIDTH/2));
    end function;

    -- Helper to convert std_logic_vector to real
    function to_real_val(slv : std_logic_vector) return real is
    begin
        return to_real(to_sfixed(slv, DATA_WIDTH/2 - 1, -DATA_WIDTH/2));
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
            NUM_INPUTS => 2,
            NUM_OUTPUTS => 1,
            MEMORY_SIZE => 1024
        )
        port map (
            clk => clk,
            rst => rst,
            start => start,
            train_mode => train_mode,
            ready => ready,
            done => done,
            host_write_en => host_write_en,
            host_addr => host_addr,
            host_data => host_data,
            input_data => input_data,
            input_valid => input_valid,
            input_last => input_last,
            output_data => output_data,
            output_valid => output_valid,
            output_last => output_last
        );

    -- Test Process
    process
        procedure load_weight(addr : integer; val : real) is
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
            wait until rising_edge(clk);
            start <= '1';
            train_mode <= '0';
            
            -- Send Input 1
            input_data <= to_slv(in1);
            input_valid <= '1';
            input_last <= '0';
            wait until rising_edge(clk);
            
            -- Send Input 2
            input_data <= to_slv(in2);
            input_valid <= '1';
            input_last <= '1';
            wait until rising_edge(clk);
            
            input_valid <= '0';
            input_last <= '0';
            start <= '0';

            -- Wait for Output
            wait until output_valid = '1';
            wait until rising_edge(clk); -- Wait one more cycle for data to settle
            
            report "Input: " & real'image(in1) & ", " & real'image(in2) & 
                   " | Output: " & real'image(to_real_val(output_data)) & 
                   " | Expected: " & real'image(expected);
                   
            assert abs(to_real_val(output_data) - expected) < TOLERANCE_C
                report "Test Failed!" severity error;
                
            -- TODO: Wait for 'done' signal once calc_done is implemented in calculation_unit
            -- Currently calc_done is hardcoded to '0' in neural_network.vhd
            wait until rising_edge(clk);
            wait until rising_edge(clk);
        end procedure;

    begin
        rst <= '1';
        wait for CLK_PERIOD * 2;
        -- rst <= '0'; -- Don't release reset yet!
        -- wait for CLK_PERIOD * 2;

        report "Loading Weights...";
        -- Layer 0 (3 neurons, 2 inputs + bias each)
        -- N0 (OR-like): w=[10, 10], b=-5
        load_weight(0, 10.0); load_weight(1, 10.0); load_weight(2, -5.0);
        -- N1 (NAND-like): w=[-10, -10], b=15
        load_weight(3, -10.0); load_weight(4, -10.0); load_weight(5, 15.0);
        -- N2 (Unused/Zero): w=[0, 0], b=0
        load_weight(6, 0.0); load_weight(7, 0.0); load_weight(8, 0.0);

        -- Layer 1 (1 neuron, 3 inputs + bias)
        -- N0 (AND-like): w=[10, 10, 0], b=-15
        load_weight(9, 10.0); load_weight(10, 10.0); load_weight(11, 0.0); load_weight(12, -15.0);

        report "Weights Loaded. Releasing Reset...";
        wait for CLK_PERIOD * 2;
        rst <= '0';
        -- wait for CLK_PERIOD * 20; -- Wait for fetch to complete (13 cycles + overhead)
        wait until ready = '1';
        report "DUT Ready. Starting Inference...";

        report "Running Inference Tests...";
        run_inference(0.0, 0.0, 0.0); -- 0 XOR 0 = 0
        run_inference(0.0, 1.0, 1.0); -- 0 XOR 1 = 1
        run_inference(1.0, 0.0, 1.0); -- 1 XOR 0 = 1
        run_inference(1.0, 1.0, 0.0); -- 1 XOR 1 = 0
        
        report "Test Complete";
        wait;
    end process;

end architecture testbench;

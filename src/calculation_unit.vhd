library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;

entity calculation_unit is
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Data Flow Control
        mode : in std_logic; -- '0' = Forward, '1' = Backward
        start_store : in std_logic; -- Trigger to write gradients to memory

        -- Network Inputs (Forward)
        input_data  : in std_logic_vector(DATA_WIDTH - 1 downto 0); -- Streaming scalar? 
                                                                    -- XOR TB sends scalar stream.
                                                                    -- We need to buffer to vector.
        input_valid : in std_logic;
        input_last  : in std_logic;

        -- Network Outputs (Forward)
        output_data  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_valid : out std_logic;
        output_last  : out std_logic;

        -- Network Error Inputs (Backward)
        error_in       : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        error_in_valid : in std_logic;

        -- Memory Interface
        mem_read_req   : out std_logic;
        mem_read_addr  : out integer;
        mem_read_data  : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        mem_read_valid : in std_logic;

        mem_update_en   : out std_logic;
        mem_update_addr : out integer;
        mem_update_grad : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        
        ready : out std_logic -- Indicates weights are loaded
    );
end entity calculation_unit;

architecture rtl of calculation_unit is

    -- Topology: 2 inputs -> Layer 0 (3 neurons) -> Layer 1 (1 neuron) -> Output
    
    -- Signals
    signal l0_fwd_ctrl : layer_control_t;
    signal l0_fwd_data : std_logic_bus_array(0 to 1)(DATA_WIDTH - 1 downto 0); -- 2 inputs
    signal l0_out_ctrl : layer_control_t;
    signal l0_out_data : std_logic_bus_array(0 to 2)(DATA_WIDTH - 1 downto 0); -- 3 outputs

    signal l1_out_ctrl : layer_control_t;
    signal l1_out_data : std_logic_bus_array(0 to 0)(DATA_WIDTH - 1 downto 0); -- 1 output

    -- Weight Fetching State Machine
    type state_t is (IDLE, FETCH_WEIGHTS, WEIGHTS_READY, STORE_GRADIENTS);
    signal state : state_t := IDLE;
    signal fetch_addr : integer := 0;
    signal fetch_layer : integer := 0; -- 0 for L0, 1 for L1
    signal fetch_idx : integer := 0;   -- Index within layer weights
    
    constant L0_WEIGHT_COUNT : integer := (2+1)*3;
    constant L1_WEIGHT_COUNT : integer := (3+1)*1;

    signal l0_weights : std_logic_bus_array(0 to L0_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    signal l1_weights : std_logic_bus_array(0 to L1_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);

    -- Gradient Buffer
    signal l0_grad_weights : std_logic_bus_array(0 to L0_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    signal l1_grad_weights : std_logic_bus_array(0 to L1_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    signal store_idx : integer := 0;
    signal store_layer : integer := 0;


begin

    -- Input Buffering (Stream -> Parallel)
    process(clk)
        variable idx : integer := 0;
    begin
        if rising_edge(clk) then
            report "Calc Unit Debug: input_valid=" & std_logic'image(input_valid) & 
                   " l0_valid=" & std_logic'image(l0_fwd_ctrl.valid) & " l0_last=" & std_logic'image(l0_fwd_ctrl.last) &
                   " l1_valid=" & std_logic'image(l1_out_ctrl.valid) & " l1_last=" & std_logic'image(l1_out_ctrl.last);
            
            if input_valid = '1' then
                l0_fwd_data(idx) <= input_data;
                if idx = 1 then
                    idx := 0;
                    l0_fwd_ctrl.valid <= '1';
                    l0_fwd_ctrl.last <= input_last; -- Propagate last signal
                else
                    idx := idx + 1;
                    l0_fwd_ctrl.valid <= '0';
                    l0_fwd_ctrl.last <= '0';
                end if;
            else
                l0_fwd_ctrl.valid <= '0';
                l0_fwd_ctrl.last <= '0';
            end if;
        end if;
    end process;

    -- Weight Fetching Logic
    process(clk, rst)
    begin
        if rst = '1' then
            state <= IDLE;
            mem_read_req <= '0';
            fetch_addr <= 0;
            l0_weights <= (others => (others => '0'));
            l1_weights <= (others => (others => '0'));
            ready <= '0';
        elsif rising_edge(clk) then
            -- Debug State Machine
            if state /= IDLE or mem_read_valid = '1' then
                 report "Calc State: " & state_t'image(state) & 
                        " ReadReq=" & std_logic'image(mem_read_req) & 
                        " ReadValid=" & std_logic'image(mem_read_valid);
            end if;

            case state is
                when IDLE =>
                    ready <= '0';
                    -- Auto-load weights on reset/start (simplified)
                    state <= FETCH_WEIGHTS;
                    fetch_addr <= 0;
                    fetch_layer <= 0;
                    fetch_idx <= 0;
                    mem_read_req <= '1';
                    mem_read_addr <= 0;
                    
                when FETCH_WEIGHTS =>
                    ready <= '0';
                    mem_read_req <= '0'; -- Pulse request
                    if mem_read_valid = '1' then
                        report "Fetch Debug: L=" & integer'image(fetch_layer) & 
                               " idx=" & integer'image(fetch_idx) & 
                               " addr=" & integer'image(fetch_addr) & -- This is NEXT addr, not current data addr
                               " data=" & to_hstring(mem_read_data);
                               
                        -- Store received weight
                        if fetch_layer = 0 then
                            l0_weights(fetch_idx) <= mem_read_data;
                            if fetch_idx = L0_WEIGHT_COUNT - 1 then
                                fetch_layer <= 1;
                                fetch_idx <= 0;
                            else
                                fetch_idx <= fetch_idx + 1;
                            end if;
                        elsif fetch_layer = 1 then
                            l1_weights(fetch_idx) <= mem_read_data;
                            if fetch_idx = L1_WEIGHT_COUNT - 1 then
                                state <= WEIGHTS_READY;
                                mem_read_req <= '0';
                            else
                                fetch_idx <= fetch_idx + 1;
                            end if;
                        end if;
                        
                        -- Request next address (if not done)
                        if state = FETCH_WEIGHTS and not (fetch_layer = 1 and fetch_idx = L1_WEIGHT_COUNT - 1) then
                            fetch_addr <= fetch_addr + 1;
                            mem_read_addr <= fetch_addr + 1;
                            mem_read_req <= '1';
                        end if;
                    end if;
                    
                when WEIGHTS_READY =>
                    mem_read_req <= '0';
                    ready <= '1';
                    
                    if start_store = '1' then
                        state <= STORE_GRADIENTS;
                        store_layer <= 0;
                        store_idx <= 0;
                        ready <= '0';
                    end if;

                when STORE_GRADIENTS =>
                    ready <= '0';
                    mem_update_en <= '1';
                    
                    if store_layer = 0 then
                        mem_update_addr <= store_idx; -- Assuming L0 weights start at 0
                        mem_update_grad <= l0_grad_weights(store_idx);
                        
                        if store_idx = L0_WEIGHT_COUNT - 1 then
                            store_layer <= 1;
                            store_idx <= 0;
                        else
                            store_idx <= store_idx + 1;
                        end if;
                    elsif store_layer = 1 then
                        mem_update_addr <= L0_WEIGHT_COUNT + store_idx; -- Offset for L1
                        mem_update_grad <= l1_grad_weights(store_idx);
                        
                        if store_idx = L1_WEIGHT_COUNT - 1 then
                            state <= WEIGHTS_READY;
                            mem_update_en <= '0';
                            store_layer <= 0;
                            store_idx <= 0;
                        else
                            store_idx <= store_idx + 1;
                        end if;
                    end if;
            end case;
        end if;
    end process;

    -- Layer 0 (Hidden: 2 -> 3)
    u_l0: entity work.layer
        generic map (NUM_INPUTS => 2, LAYER_SIZE => 3)
        port map (
            clk => clk, rst => rst,
            fwd_en => not mode, bwd_en => mode,
            fwd_ctrl_in => l0_fwd_ctrl, fwd_data_in => l0_fwd_data,
            fwd_ctrl_out => l0_out_ctrl, fwd_data_out => l0_out_data,
            bwd_ctrl_in => (valid=>'0', last=>'0'), bwd_error_in => (others=>(others=>'0')),
            weights_in => l0_weights
            -- grads_out => ...
        );

    -- Layer 1 (Output: 3 -> 1)
    u_l1: entity work.layer
        generic map (NUM_INPUTS => 3, LAYER_SIZE => 1)
        port map (
            clk => clk, rst => rst,
            fwd_en => not mode, bwd_en => mode,
            fwd_ctrl_in => l0_out_ctrl, fwd_data_in => l0_out_data,
            fwd_ctrl_out => l1_out_ctrl, fwd_data_out => l1_out_data,
            bwd_ctrl_in => (valid=>'0', last=>'0'), bwd_error_in => (others=>(others=>'0')),
            weights_in => l1_weights
        );

    -- Output
    output_valid <= l1_out_ctrl.valid;
    output_last <= l1_out_ctrl.last;
    output_data <= l1_out_data(0);





end architecture rtl;

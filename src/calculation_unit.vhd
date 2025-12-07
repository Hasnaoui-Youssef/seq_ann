library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;

entity calculation_unit is
    generic (
        NUM_INPUTS  : integer := 2;  -- Network input size
        NUM_LAYERS  : integer := 2;  -- Number of layers
        LAYER_SIZES : layer_config_array  -- Array of layer sizes (e.g., (3, 1) for 2->3->1)
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Control Interface
        start       : in std_logic;  -- Start inference/training
        mode        : in std_logic;  -- '0' = Forward, '1' = Backward
        start_store : in std_logic;  -- Trigger to write gradients to memory

        -- Status
        ready : out std_logic;  -- Ready to accept new start
        done  : out std_logic;  -- Inference/backward pass complete

        -- Network Outputs (Forward)
        output_data  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_valid : out std_logic;

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
        mem_update_grad : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity calculation_unit;

architecture rtl of calculation_unit is

    -- Calculate total weights needed for all layers
    function calc_total_weights return integer is
        variable total : integer := 0;
        variable prev_size : integer := NUM_INPUTS;
    begin
        for i in LAYER_SIZES'range loop
            total := total + (prev_size + 1) * LAYER_SIZES(i);  -- weights + bias per neuron
            prev_size := LAYER_SIZES(i);
        end loop;
        return total;
    end function;

    -- Calculate weight offset for a specific layer
    function calc_layer_weight_offset(layer_idx : integer) return integer is
        variable offset : integer := 0;
        variable prev_size : integer := NUM_INPUTS;
    begin
        for i in 0 to layer_idx - 1 loop
            offset := offset + (prev_size + 1) * LAYER_SIZES(i);
            prev_size := LAYER_SIZES(i);
        end loop;
        return offset;
    end function;

    -- Get input size for a layer
    function get_layer_input_size(layer_idx : integer) return integer is
    begin
        if layer_idx = 0 then
            return NUM_INPUTS;
        else
            return LAYER_SIZES(layer_idx - 1);
        end if;
    end function;

    -- Maximum layer size (for signal array sizing)
    function max_layer_size return integer is
        variable max_size : integer := NUM_INPUTS;
    begin
        for i in LAYER_SIZES'range loop
            if LAYER_SIZES(i) > max_size then
                max_size := LAYER_SIZES(i);
            end if;
        end loop;
        return max_size;
    end function;

    -- Constants
    constant TOTAL_WEIGHTS : integer := calc_total_weights;
    constant OUTPUT_SIZE   : integer := LAYER_SIZES(NUM_LAYERS - 1);
    constant MAX_SIZE      : integer := max_layer_size;
    
    -- Memory layout: inputs at 0..NUM_INPUTS-1, weights at NUM_INPUTS..NUM_INPUTS+TOTAL_WEIGHTS-1
    constant WEIGHT_BASE_ADDR : integer := NUM_INPUTS;

    -- Inter-layer signals (control and data between layers)
    type ctrl_array_t is array (0 to NUM_LAYERS) of layer_control_t;
    type data_array_t is array (0 to NUM_LAYERS) of std_logic_bus_array(0 to MAX_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    signal fwd_ctrl : ctrl_array_t;
    signal fwd_data : data_array_t;
    signal bwd_ctrl : ctrl_array_t;
    signal bwd_data : data_array_t;

    -- Input storage (fetched from memory)
    signal input_buffer : std_logic_bus_array(0 to NUM_INPUTS - 1)(DATA_WIDTH - 1 downto 0);
    
    -- Weight storage (flat array for all layers)
    signal all_weights : std_logic_bus_array(0 to TOTAL_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);
    signal all_grads   : std_logic_bus_array(0 to TOTAL_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

    -- State Machine
    type state_t is (
        IDLE,               -- Ready for start signal
        FETCH_INPUTS_REQ,   -- Request input from memory
        FETCH_INPUTS_WAIT,  -- Wait for input data
        FETCH_WEIGHTS_REQ,  -- Request weight from memory
        FETCH_WEIGHTS_WAIT, -- Wait for weight data
        FORWARD_START,      -- Trigger forward pass
        FORWARD_WAIT,       -- Wait for forward pass completion
        STORE_GRADIENTS,    -- Store gradients to memory (training)
        DONE_STATE          -- Signal completion
    );
    signal state : state_t := IDLE;
    signal fetch_idx : integer := 0;
    signal store_idx : integer := 0;
    
    -- Forward pass control
    signal fwd_trigger : std_logic := '0';
    signal fwd_complete : std_logic := '0';

begin

    -- Forward pass completion detection
    -- The forward pass is complete when output_valid goes high
    fwd_complete <= fwd_ctrl(NUM_LAYERS).valid;

    -- Main State Machine
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state <= IDLE;
                mem_read_req <= '0';
                mem_update_en <= '0';
                fetch_idx <= 0;
                store_idx <= 0;
                ready <= '0';
                done <= '0';
                fwd_trigger <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                input_buffer <= (others => (others => '0'));
                all_weights <= (others => (others => '0'));
            else
                -- Default assignments
                mem_read_req <= '0';
                mem_update_en <= '0';
                fwd_trigger <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                
                case state is
                    when IDLE =>
                        ready <= '1';
                        done <= '0';
                        if start = '1' then
                            ready <= '0';
                            fetch_idx <= 0;
                            state <= FETCH_INPUTS_REQ;
                        end if;

                    -- Fetch inputs from memory (addresses 0 to NUM_INPUTS-1)
                    when FETCH_INPUTS_REQ =>
                        mem_read_addr <= fetch_idx;
                        mem_read_req <= '1';
                        state <= FETCH_INPUTS_WAIT;

                    when FETCH_INPUTS_WAIT =>
                        if mem_read_valid = '1' then
                            input_buffer(fetch_idx) <= mem_read_data;
                            if fetch_idx = NUM_INPUTS - 1 then
                                fetch_idx <= 0;
                                state <= FETCH_WEIGHTS_REQ;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= FETCH_INPUTS_REQ;
                            end if;
                        end if;

                    -- Fetch weights from memory (addresses NUM_INPUTS to NUM_INPUTS+TOTAL_WEIGHTS-1)
                    when FETCH_WEIGHTS_REQ =>
                        mem_read_addr <= WEIGHT_BASE_ADDR + fetch_idx;
                        mem_read_req <= '1';
                        state <= FETCH_WEIGHTS_WAIT;

                    when FETCH_WEIGHTS_WAIT =>
                        if mem_read_valid = '1' then
                            all_weights(fetch_idx) <= mem_read_data;
                            if fetch_idx = TOTAL_WEIGHTS - 1 then
                                state <= FORWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= FETCH_WEIGHTS_REQ;
                            end if;
                        end if;

                    -- Trigger forward pass by asserting valid on first layer input
                    when FORWARD_START =>
                        -- Load inputs to first layer
                        fwd_data(0)(0 to NUM_INPUTS - 1) <= input_buffer;
                        fwd_ctrl(0).valid <= '1';
                        fwd_ctrl(0).last <= '1';  -- Single sample
                        fwd_trigger <= '1';
                        state <= FORWARD_WAIT;

                    when FORWARD_WAIT =>
                        -- Wait for output to be valid
                        if fwd_complete = '1' then
                            if mode = '1' and start_store = '1' then
                                -- Training mode: store gradients
                                store_idx <= 0;
                                state <= STORE_GRADIENTS;
                            else
                                -- Inference mode: done
                                state <= DONE_STATE;
                            end if;
                        end if;

                    when STORE_GRADIENTS =>
                        mem_update_en <= '1';
                        mem_update_addr <= WEIGHT_BASE_ADDR + store_idx;
                        mem_update_grad <= all_grads(store_idx);
                        if store_idx = TOTAL_WEIGHTS - 1 then
                            mem_update_en <= '0';
                            state <= DONE_STATE;
                        else
                            store_idx <= store_idx + 1;
                        end if;

                    when DONE_STATE =>
                        done <= '1';
                        ready <= '1';
                        if start = '1' then
                            done <= '0';
                            ready <= '0';
                            fetch_idx <= 0;
                            state <= FETCH_INPUTS_REQ;
                        end if;

                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    -- Generate Layers
    gen_layers: for i in 0 to NUM_LAYERS - 1 generate
        constant THIS_INPUT_SIZE  : integer := get_layer_input_size(i);
        constant THIS_OUTPUT_SIZE : integer := LAYER_SIZES(i);
        constant THIS_WEIGHT_COUNT: integer := (THIS_INPUT_SIZE + 1) * THIS_OUTPUT_SIZE;
        constant WEIGHT_OFFSET    : integer := calc_layer_weight_offset(i);

        signal layer_weights : std_logic_bus_array(0 to THIS_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
        signal layer_grads   : std_logic_bus_array(0 to THIS_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    begin
        -- Map weights from flat array to layer
        gen_weight_map: for w in 0 to THIS_WEIGHT_COUNT - 1 generate
            layer_weights(w) <= all_weights(WEIGHT_OFFSET + w);
            all_grads(WEIGHT_OFFSET + w) <= layer_grads(w);
        end generate;

        -- Layer instance
        u_layer: entity work.layer
            generic map (
                NUM_INPUTS => THIS_INPUT_SIZE,
                LAYER_SIZE => THIS_OUTPUT_SIZE,
                USE_SIGMOID => true
            )
            port map (
                clk => clk,
                rst => rst,
                fwd_en => not mode,
                bwd_en => mode,
                fwd_ctrl_in => fwd_ctrl(i),
                fwd_data_in => fwd_data(i)(0 to THIS_INPUT_SIZE - 1),
                fwd_ctrl_out => fwd_ctrl(i + 1),
                fwd_data_out => fwd_data(i + 1)(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_in => bwd_ctrl(i + 1),
                bwd_error_in => bwd_data(i + 1)(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_out => bwd_ctrl(i),
                bwd_error_out => bwd_data(i)(0 to THIS_INPUT_SIZE - 1),
                weights_in => layer_weights,
                grads_out => layer_grads
            );
    end generate;

    -- Output from last layer
    output_valid <= fwd_ctrl(NUM_LAYERS).valid;
    output_data  <= fwd_data(NUM_LAYERS)(0);

    -- Initialize backward path (no error from output for now)
    bwd_ctrl(NUM_LAYERS).valid <= '0';
    bwd_ctrl(NUM_LAYERS).last <= '0';
    bwd_data(NUM_LAYERS) <= (others => (others => '0'));

end architecture rtl;

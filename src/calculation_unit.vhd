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
        load_weights : in std_logic;  -- Start weight loading from memory
        start        : in std_logic;  -- Start inference (weights must be loaded)
        mode         : in std_logic;  -- '0' = Forward, '1' = Backward
        start_store  : in std_logic;  -- Trigger to write gradients to memory

        -- Status
        weights_loaded : out std_logic;  -- Weights are loaded and ready
        ready          : out std_logic;  -- Ready to accept new start
        done           : out std_logic;  -- Inference/backward pass complete

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

    -- Weight loading signals (directly to layers)
    signal layer_weight_load_en   : std_logic_vector(0 to NUM_LAYERS - 1) := (others => '0');
    signal layer_weight_load_done : std_logic_vector(0 to NUM_LAYERS - 1);

    -- State Machine
    type state_t is (
        IDLE,                   -- Ready for commands
        LOAD_WEIGHTS_REQ,       -- Request weight from memory
        LOAD_WEIGHTS_WAIT,      -- Wait for weight data
        LOAD_WEIGHTS_SEND,      -- Send weight to layer
        WEIGHTS_READY,          -- Weights loaded, ready for inference
        FETCH_INPUTS_REQ,       -- Request input from memory
        FETCH_INPUTS_WAIT,      -- Wait for input data
        FORWARD_START,          -- Trigger forward pass
        FORWARD_WAIT,           -- Wait for forward pass completion
        STORE_GRADIENTS,        -- Store gradients to memory (training)
        DONE_STATE              -- Signal completion
    );
    signal state : state_t := IDLE;

    -- Counters and indices
    signal fetch_idx : integer := 0;
    signal store_idx : integer := 0;
    signal current_layer : integer range 0 to NUM_LAYERS - 1 := 0;
    signal layer_weight_idx : integer := 0;  -- Index within current layer

    -- Status registers
    signal weights_loaded_reg : std_logic := '0';

    -- Forward pass control
    signal fwd_complete : std_logic := '0';

begin

    -- Forward pass completion detection
    fwd_complete <= fwd_ctrl(NUM_LAYERS).valid;

    -- Status outputs
    weights_loaded <= weights_loaded_reg;

    -- Main State Machine
    process(clk)
        variable layer_weight_count : integer;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state <= IDLE;
                mem_read_req <= '0';
                mem_read_addr <= 0;
                mem_update_en <= '0';
                mem_update_addr <= 0;
                mem_update_grad <= (others => '0');
                fetch_idx <= 0;
                store_idx <= 0;
                current_layer <= 0;
                layer_weight_idx <= 0;
                ready <= '0';
                done <= '0';
                weights_loaded_reg <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                input_buffer <= (others => (others => '0'));
                layer_weight_load_en <= (others => '0');
            else
                -- Default assignments
                mem_read_req <= '0';
                mem_update_en <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                layer_weight_load_en <= (others => '0');

                case state is
                    when IDLE =>
                        ready <= '0';
                        done <= '0';
                        if load_weights = '1' then
                            -- Start weight loading
                            fetch_idx <= 0;
                            current_layer <= 0;
                            layer_weight_idx <= 0;
                            weights_loaded_reg <= '0';
                            state <= LOAD_WEIGHTS_REQ;
                        elsif start = '1' and weights_loaded_reg = '1' then
                            -- Start inference (only if weights loaded)
                            fetch_idx <= 0;
                            state <= FETCH_INPUTS_REQ;
                        end if;

                    -- Load weights from memory to layer weight banks
                    when LOAD_WEIGHTS_REQ =>
                        mem_read_addr <= WEIGHT_BASE_ADDR + fetch_idx;
                        mem_read_req <= '1';
                        state <= LOAD_WEIGHTS_WAIT;

                    when LOAD_WEIGHTS_WAIT =>
                        if mem_read_valid = '1' then
                            state <= LOAD_WEIGHTS_SEND;
                        end if;

                    when LOAD_WEIGHTS_SEND =>
                        -- Send weight to current layer's weight bank
                        layer_weight_load_en(current_layer) <= '1';

                        if current_layer = 0 then
                            layer_weight_count := (NUM_INPUTS + 1) * LAYER_SIZES(0);
                        else
                            layer_weight_count := (LAYER_SIZES(current_layer - 1) + 1) * LAYER_SIZES(current_layer);
                        end if;

                        if layer_weight_idx = layer_weight_count - 1 then
                            -- Done with this layer
                            layer_weight_idx <= 0;
                            if current_layer = NUM_LAYERS - 1 then
                                -- All layers loaded
                                weights_loaded_reg <= '1';
                                state <= WEIGHTS_READY;
                            else
                                -- Move to next layer
                                current_layer <= current_layer + 1;
                                fetch_idx <= fetch_idx + 1;
                                state <= LOAD_WEIGHTS_REQ;
                            end if;
                        else
                            layer_weight_idx <= layer_weight_idx + 1;
                            fetch_idx <= fetch_idx + 1;
                            state <= LOAD_WEIGHTS_REQ;
                        end if;

                    when WEIGHTS_READY =>
                        ready <= '1';
                        if start = '1' then
                            ready <= '0';
                            fetch_idx <= 0;
                            state <= FETCH_INPUTS_REQ;
                        elsif load_weights = '1' then
                            -- Allow reloading weights
                            ready <= '0';
                            fetch_idx <= 0;
                            current_layer <= 0;
                            layer_weight_idx <= 0;
                            weights_loaded_reg <= '0';
                            state <= LOAD_WEIGHTS_REQ;
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
                                state <= FORWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= FETCH_INPUTS_REQ;
                            end if;
                        end if;

                    -- Trigger forward pass
                    when FORWARD_START =>
                        fwd_data(0)(0 to NUM_INPUTS - 1) <= input_buffer;
                        fwd_ctrl(0).valid <= '1';
                        fwd_ctrl(0).last <= '1';
                        state <= FORWARD_WAIT;

                    when FORWARD_WAIT =>
                        if fwd_complete = '1' then
                            if mode = '1' and start_store = '1' then
                                store_idx <= 0;
                                state <= STORE_GRADIENTS;
                            else
                                state <= DONE_STATE;
                            end if;
                        end if;

                    when STORE_GRADIENTS =>
                        -- TODO: Implement gradient storage
                        state <= DONE_STATE;

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

    -- Generate Layers with weight bank interface
    gen_layers: for i in 0 to NUM_LAYERS - 1 generate
        constant THIS_INPUT_SIZE  : integer := get_layer_input_size(i);
        constant THIS_OUTPUT_SIZE : integer := LAYER_SIZES(i);
        constant THIS_WEIGHT_COUNT: integer := (THIS_INPUT_SIZE + 1) * THIS_OUTPUT_SIZE;

        signal layer_grads : std_logic_bus_array(0 to THIS_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    begin
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
                -- Weight bank interface
                weight_load_en => layer_weight_load_en(i),
                weight_load_data => mem_read_data,
                weight_load_done => layer_weight_load_done(i),
                weight_save_en => '0',
                weight_save_data => open,
                weight_save_done => open,
                weight_update_en => '0',
                weight_learn_rate => (others => '0'),
                weight_update_done => open,
                grads_out => layer_grads
            );
    end generate;

    -- Output from last layer
    output_valid <= fwd_ctrl(NUM_LAYERS).valid;
    output_data  <= fwd_data(NUM_LAYERS)(0);

    -- Initialize backward path
    bwd_ctrl(NUM_LAYERS).valid <= '0';
    bwd_ctrl(NUM_LAYERS).last <= '0';
    bwd_data(NUM_LAYERS) <= (others => (others => '0'));

end architecture rtl;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;
use IEEE.fixed_pkg.all;

entity calculation_unit is
    generic (
        NUM_INPUTS  : integer := 2;
        NUM_LAYERS  : integer := 2;
        LAYER_SIZES : layer_config_array;
        USE_SIGMOID : boolean_array
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Control Interface
        load_weights : in std_logic;
        start        : in std_logic;
        mode         : in std_logic;  -- '0' = Forward only, '1' = Forward + Backward + Update
        start_store  : in std_logic;

        -- Status
        weights_loaded : out std_logic;
        ready          : out std_logic;
        done           : out std_logic;

        -- Network Outputs (Forward) - sized by last layer
        output_data  : out sfixed_bus_array(0 to LAYER_SIZES(LAYER_SIZES'high) - 1);
        output_valid : out std_logic;

        -- Training parameters
        learning_rate : in sfixed_bus;

        -- Memory Interface (still std_logic_vector for BRAM compatibility)
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

    function get_layer_input_size(layer_idx : integer) return integer is
    begin
        if layer_idx = 0 then
            return NUM_INPUTS;
        else
            return LAYER_SIZES(layer_idx - 1);
        end if;
    end function;

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

    -- Error computation: output - target (for MSE loss gradient)
    function compute_error(
        output_val : sfixed_bus;
        target_val : sfixed_bus
    ) return sfixed_bus is
    begin
        return resize(output_val - target_val, INT_BITS - 1, -FRAC_BITS);
    end function;

    constant MAX_SIZE      : integer := max_layer_size;
    constant NUM_OUTPUTS   : integer := LAYER_SIZES(LAYER_SIZES'high);

    -- Compute total weight count across all layers for memory layout
    function total_weight_count return integer is
        variable total : integer := 0;
        variable layer_inputs : integer;
    begin
        for i in LAYER_SIZES'range loop
            if i = LAYER_SIZES'low then
                layer_inputs := NUM_INPUTS;
            else
                layer_inputs := LAYER_SIZES(i - 1);
            end if;
            total := total + (layer_inputs + 1) * LAYER_SIZES(i);
        end loop;
        return total;
    end function;

    constant TOTAL_WEIGHTS : integer := total_weight_count;

    -- Memory layout:
    --   [0..NUM_INPUTS-1]                                      : Input data
    --   [NUM_INPUTS..NUM_INPUTS+NUM_OUTPUTS-1]                 : Target data (training)
    --   [NUM_INPUTS+NUM_OUTPUTS..NUM_INPUTS+NUM_OUTPUTS+TW-1]  : Weights
    constant TARGET_BASE_ADDR : integer := NUM_INPUTS;
    constant WEIGHT_BASE_ADDR : integer := NUM_INPUTS + NUM_OUTPUTS;

    type ctrl_array_t is array (0 to NUM_LAYERS) of layer_control_t;
    type data_array_t is array (0 to NUM_LAYERS) of sfixed_bus_array(0 to MAX_SIZE - 1);

    constant CTRL_INIT : layer_control_t := (valid => '0', last => '0');
    constant DATA_INIT : sfixed_bus_array(0 to MAX_SIZE - 1) := (others => (others => '0'));

    signal fwd_ctrl : ctrl_array_t := (others => CTRL_INIT);
    signal fwd_data : data_array_t := (others => DATA_INIT);
    signal bwd_ctrl : ctrl_array_t := (others => CTRL_INIT);
    signal bwd_data : data_array_t := (others => DATA_INIT);

    signal input_buffer : sfixed_bus_array(0 to NUM_INPUTS - 1)
        := (others => (others => '0'));

    signal layer_weight_load_en   : std_logic_vector(0 to NUM_LAYERS - 1) := (others => '0');
    signal layer_weight_load_done : std_logic_vector(0 to NUM_LAYERS - 1);

    -- Latched mode signal to prevent mid-computation corruption
    signal mode_latched : std_logic := '0';

    type state_t is (
        IDLE,                   -- Ready for commands
        LOAD_WEIGHTS_REQ,       -- Request weight from memory
        LOAD_WEIGHTS_WAIT,      -- Wait for weight data and send to weight bank
        WEIGHTS_READY,          -- Weights loaded, ready for inference
        FETCH_INPUTS_REQ,       -- Request input from memory
        FETCH_INPUTS_WAIT,      -- Wait for input data
        FORWARD_START,          -- Trigger forward pass
        FORWARD_WAIT,           -- Wait for forward pass completion
        LOAD_TARGET_REQ,        -- Request target from memory (training)
        LOAD_TARGET_WAIT,       -- Wait for target data
        BACKWARD_START,         -- Inject error and trigger backward pass
        BACKWARD_WAIT,          -- Wait for backward pass completion
        UPDATE_WEIGHTS,         -- Trigger weight bank updates
        UPDATE_WEIGHTS_WAIT,    -- Wait for weight updates to complete
        DONE_STATE              -- Signal completion
    );
    signal state : state_t := IDLE;

    -- Counters and indices
    signal fetch_idx : integer := 0;
    signal current_layer : integer range 0 to NUM_LAYERS - 1 := 0;
    signal layer_weight_idx : integer := 0;

    signal weights_loaded_reg : std_logic := '0';

    signal fwd_complete : std_logic := '0';

    signal output_data_reg : sfixed_bus_array(0 to NUM_OUTPUTS - 1)
        := (others => (others => '0'));
    signal output_valid_reg : std_logic := '0';

    -- Target buffer for training
    signal target_buffer : sfixed_bus_array(0 to NUM_OUTPUTS - 1)
        := (others => (others => '0'));

    -- Weight update control signals
    signal layer_weight_update_en   : std_logic_vector(0 to NUM_LAYERS - 1) := (others => '0');
    signal layer_weight_update_done : std_logic_vector(0 to NUM_LAYERS - 1);

    -- Backward pass completion
    signal bwd_complete : std_logic := '0';

    -- Phase control: during training, forward phase first, then backward phase
    signal in_backward_phase : std_logic := '0';

    -- BRAM data converted to sfixed at boundary
    signal mem_read_sfixed : sfixed_bus;

begin

    mem_read_sfixed <= to_sfixed(mem_read_data, INT_BITS - 1, -FRAC_BITS);

    fwd_complete <= fwd_ctrl(NUM_LAYERS).valid;
    bwd_complete <= bwd_ctrl(0).valid;

    weights_loaded <= weights_loaded_reg;

    process(state, mem_read_valid, current_layer)
    begin
        layer_weight_load_en <= (others => '0');
        if state = LOAD_WEIGHTS_WAIT and mem_read_valid = '1' then
            layer_weight_load_en(current_layer) <= '1';
        end if;
    end process;

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
                current_layer <= 0;
                layer_weight_idx <= 0;
                ready <= '0';
                done <= '0';
                weights_loaded_reg <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                bwd_ctrl(NUM_LAYERS).valid <= '0';
                bwd_ctrl(NUM_LAYERS).last <= '0';
                bwd_data(NUM_LAYERS) <= (others => (others => '0'));
                input_buffer <= (others => (others => '0'));
                target_buffer <= (others => (others => '0'));
                output_data_reg <= (others => (others => '0'));
                output_valid_reg <= '0';
                mode_latched <= '0';
                in_backward_phase <= '0';
            else
                -- Default assignments
                mem_read_req <= '0';
                mem_update_en <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                bwd_ctrl(NUM_LAYERS).valid <= '0';
                bwd_ctrl(NUM_LAYERS).last <= '0';

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
                            fetch_idx <= 0;
                            mode_latched <= mode;
                            state <= FETCH_INPUTS_REQ;
                        end if;

                    when LOAD_WEIGHTS_REQ =>
                        mem_read_addr <= WEIGHT_BASE_ADDR + fetch_idx;
                        mem_read_req <= '1';
                        state <= LOAD_WEIGHTS_WAIT;

                    when LOAD_WEIGHTS_WAIT =>
                        if mem_read_valid = '1' then
                            if current_layer = 0 then
                                layer_weight_count := (NUM_INPUTS + 1) * LAYER_SIZES(0);
                            else
                                layer_weight_count := (LAYER_SIZES(current_layer - 1) + 1) * LAYER_SIZES(current_layer);
                            end if;

                            if layer_weight_idx = layer_weight_count - 1 then
                                -- Done with this layer
                                layer_weight_idx <= 0;
                                if current_layer = NUM_LAYERS - 1 then
                                    weights_loaded_reg <= '1';
                                    state <= WEIGHTS_READY;
                                else
                                    current_layer <= current_layer + 1;
                                    fetch_idx <= fetch_idx + 1;
                                    state <= LOAD_WEIGHTS_REQ;
                                end if;
                            else
                                layer_weight_idx <= layer_weight_idx + 1;
                                fetch_idx <= fetch_idx + 1;
                                state <= LOAD_WEIGHTS_REQ;
                            end if;
                        end if;

                    when WEIGHTS_READY =>
                        ready <= '1';
                        if start = '1' then
                            ready <= '0';
                            done <= '0';
                            fetch_idx <= 0;
                            mode_latched <= mode;
                            state <= FETCH_INPUTS_REQ;
                        elsif load_weights = '1' then
                            ready <= '0';
                            fetch_idx <= 0;
                            current_layer <= 0;
                            layer_weight_idx <= 0;
                            weights_loaded_reg <= '0';
                            state <= LOAD_WEIGHTS_REQ;
                        end if;

                    when FETCH_INPUTS_REQ =>
                        mem_read_addr <= fetch_idx;
                        mem_read_req <= '1';
                        state <= FETCH_INPUTS_WAIT;

                    when FETCH_INPUTS_WAIT =>
                        if mem_read_valid = '1' then
                            input_buffer(fetch_idx) <= mem_read_sfixed;
                            if fetch_idx = NUM_INPUTS - 1 then
                                state <= FORWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= FETCH_INPUTS_REQ;
                            end if;
                        end if;

                    when FORWARD_START =>
                        fwd_data(0)(0 to NUM_INPUTS - 1) <= input_buffer;
                        fwd_ctrl(0).valid <= '1';
                        fwd_ctrl(0).last <= '1';
                        state <= FORWARD_WAIT;

                    when FORWARD_WAIT =>
                        if fwd_complete = '1' then
                            -- Latch all outputs from last layer
                            for i in 0 to NUM_OUTPUTS - 1 loop
                                output_data_reg(i) <= fwd_data(NUM_LAYERS)(i);
                            end loop;
                            output_valid_reg <= '1';
                            if mode_latched = '1' then
                                -- Training: load targets next
                                fetch_idx <= 0;
                                state <= LOAD_TARGET_REQ;
                            else
                                state <= DONE_STATE;
                            end if;
                        end if;

                    when LOAD_TARGET_REQ =>
                        mem_read_addr <= TARGET_BASE_ADDR + fetch_idx;
                        mem_read_req <= '1';
                        state <= LOAD_TARGET_WAIT;

                    when LOAD_TARGET_WAIT =>
                        if mem_read_valid = '1' then
                            target_buffer(fetch_idx) <= mem_read_sfixed;
                            if fetch_idx = NUM_OUTPUTS - 1 then
                                state <= BACKWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= LOAD_TARGET_REQ;
                            end if;
                        end if;

                    when BACKWARD_START =>
                        -- Compute error = output - target for each output neuron
                        -- and inject into last layer's backward path
                        in_backward_phase <= '1';
                        for i in 0 to NUM_OUTPUTS - 1 loop
                            bwd_data(NUM_LAYERS)(i) <= compute_error(
                                output_data_reg(i), target_buffer(i));
                        end loop;
                        bwd_ctrl(NUM_LAYERS).valid <= '1';
                        bwd_ctrl(NUM_LAYERS).last <= '1';
                        state <= BACKWARD_WAIT;

                    when BACKWARD_WAIT =>
                        -- Clear backward injection after one cycle
                        bwd_ctrl(NUM_LAYERS).valid <= '0';
                        bwd_ctrl(NUM_LAYERS).last <= '0';
                        if bwd_complete = '1' then
                            state <= UPDATE_WEIGHTS;
                        end if;

                    when UPDATE_WEIGHTS =>
                        -- Trigger in-place weight update on all layers
                        layer_weight_update_en <= (others => '1');
                        state <= UPDATE_WEIGHTS_WAIT;

                    when UPDATE_WEIGHTS_WAIT =>
                        layer_weight_update_en <= (others => '0');
                        -- Wait for all layers to finish updating
                        if layer_weight_update_done = (layer_weight_update_done'range => '1') then
                            state <= DONE_STATE;
                        end if;

                    when DONE_STATE =>
                        done <= '1';
                        ready <= '0';
                        in_backward_phase <= '0';
                        state <= WEIGHTS_READY;

                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    gen_layers: for i in 0 to NUM_LAYERS - 1 generate
        constant THIS_INPUT_SIZE  : integer := get_layer_input_size(i);
        constant THIS_OUTPUT_SIZE : integer := LAYER_SIZES(i);
        constant THIS_WEIGHT_COUNT: integer := (THIS_INPUT_SIZE + 1) * THIS_OUTPUT_SIZE;

        signal layer_grads : sfixed_bus_array(0 to THIS_WEIGHT_COUNT - 1);
    begin
        u_layer: entity work.layer
            generic map (
                NUM_INPUTS => THIS_INPUT_SIZE,
                LAYER_SIZE => THIS_OUTPUT_SIZE,
                USE_SIGMOID => USE_SIGMOID(i)
            )
            port map (
                clk => clk,
                rst => rst,
                fwd_en => not in_backward_phase,
                bwd_en => in_backward_phase,
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
                weight_load_data => mem_read_sfixed,
                weight_load_done => layer_weight_load_done(i),
                weight_save_en => '0',
                weight_save_data => open,
                weight_save_done => open,
                weight_update_en => layer_weight_update_en(i),
                weight_learn_rate => learning_rate,
                weight_update_done => layer_weight_update_done(i),
                grads_out => layer_grads
            );
    end generate;

    output_valid <= output_valid_reg;
    output_data  <= output_data_reg;

end architecture rtl;

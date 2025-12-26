library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity control_unit is
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- User Interface
        start       : in std_logic;
        train_mode  : in std_logic; -- '1' = Train, '0' = Predict
        ready       : out std_logic;
        done        : out std_logic;

        -- Calculation Unit Control
        calc_mode   : out std_logic; -- '0' = Fwd, '1' = Bwd
        calc_start  : out std_logic;
        calc_store  : out std_logic;
        calc_done   : in std_logic;
        calc_ready  : in std_logic; -- New input

        -- Memory Unit Control
        learning_rate : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity control_unit;

architecture rtl of control_unit is

    type state_t is (IDLE, TRAIN_EPOCH_START, TRAIN_SAMPLE_START, TRAIN_FWD, TRAIN_BWD, TRAIN_UPDATE, PREDICT_FWD, FINISHED);
    signal state : state_t := IDLE;

    constant DEFAULT_LR : std_logic_vector(DATA_WIDTH - 1 downto 0) := x"00001999"; -- 0.1 in 16.16 fixed point (approx)

    -- Training Loop Counters (Hardcoded for now, should be configurable)
    constant MAX_EPOCHS : integer := 10;
    constant SAMPLES_PER_EPOCH : integer := 4; -- XOR has 4 samples

    signal epoch_counter : integer := 0;
    signal sample_counter : integer := 0;

begin

    learning_rate <= DEFAULT_LR;

    process(clk, rst)
    begin
        if rst = '1' then
            state <= IDLE;
            ready <= '0';
            done <= '0';
            calc_mode <= '0';
            calc_start <= '0';
            calc_store <= '0';
        elsif rising_edge(clk) then

            case state is
                when IDLE =>
                    if calc_ready = '1' then
                        ready <= '1';
                    else
                        ready <= '0';
                    end if;

                    done <= '0';
                    if start = '1' then
                        ready <= '0';
                        if train_mode = '1' then
                            state <= TRAIN_EPOCH_START;
                            epoch_counter <= 0;
                        else
                            state <= PREDICT_FWD;
                            calc_mode <= '0';
                            calc_start <= '1';
                        end if;
                    end if;

                -- Training Loop States
                when TRAIN_EPOCH_START =>
                    if epoch_counter < MAX_EPOCHS then
                        state <= TRAIN_SAMPLE_START;
                        sample_counter <= 0;
                    else
                        state <= FINISHED;
                    end if;

                when TRAIN_SAMPLE_START =>
                    if sample_counter < SAMPLES_PER_EPOCH then
                        state <= TRAIN_FWD;
                        calc_mode <= '0'; -- Forward
                        calc_start <= '1';
                    else
                        -- End of Epoch
                        epoch_counter <= epoch_counter + 1;
                        state <= TRAIN_EPOCH_START;
                    end if;

                when TRAIN_FWD =>
                    calc_start <= '0';
                    if calc_done = '1' then
                        state <= TRAIN_BWD;
                        calc_mode <= '1'; -- Backward
                        calc_start <= '1';
                    end if;

                when TRAIN_BWD =>
                    calc_start <= '0';
                    if calc_done = '1' then
                        state <= TRAIN_UPDATE;
                        calc_store <= '1'; -- Trigger gradient storage
                    end if;

                when TRAIN_UPDATE =>
                    calc_store <= '0';
                    if calc_ready = '1' then
                        -- Sample Done
                        sample_counter <= sample_counter + 1;
                        state <= TRAIN_SAMPLE_START;
                    end if;

                when PREDICT_FWD =>
                    -- After first cycle, clear calc_start
                    -- Only check calc_done after calc_start is cleared (calc has started)
                    if calc_start = '1' then
                        calc_start <= '0';
                    elsif calc_done = '1' then
                        state <= FINISHED;
                    end if;

                when FINISHED =>
                    done <= '1';
                    -- Wait for start to deassert before going to IDLE
                    if start = '0' then
                        state <= IDLE;
                    end if;

                when others =>
                    state <= IDLE;
            end case;
        end if;
    end process;

end architecture rtl;

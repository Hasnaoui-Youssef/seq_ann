library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;

-- Weight Bank: Stores weights in registers for fast parallel access
--
-- This component decouples weight loading (slow, sequential from memory)
-- from weight usage (fast, parallel to compute units).
--
-- Operation:
-- 1. LOAD: Assert load_en, provide weight on load_data. Internal counter
--    increments automatically. load_done pulses when all weights loaded.
-- 2. READ: weights_o always reflects current register contents.
-- 3. UPDATE (training): Assert update_en with grad_data and learn_rate.
--    Weights updated in place: w = w - lr * grad
-- 4. SAVE: Assert save_en, weights output sequentially on save_data.
--
-- For dense layers: NUM_WEIGHTS = (NUM_INPUTS + 1) * LAYER_SIZE
-- The +1 accounts for bias per neuron.

entity weight_bank is
    generic (
        NUM_WEIGHTS : integer := 16
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Load interface (sequential, from memory)
        load_en   : in  std_logic;
        load_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        load_done : out std_logic;
        load_idx  : out integer range 0 to NUM_WEIGHTS - 1;

        -- Read interface (parallel, to compute units)
        weights_o : out std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);

        -- Gradient update interface (training)
        update_en   : in  std_logic;
        grad_data   : in  std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0);
        learn_rate  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        update_done : out std_logic;

        -- Save interface (sequential, to memory)
        save_en   : in  std_logic;
        save_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        save_done : out std_logic;
        save_idx  : out integer range 0 to NUM_WEIGHTS - 1
    );
end entity weight_bank;

architecture rtl of weight_bank is

    -- Weight storage registers
    signal weight_regs : std_logic_bus_array(0 to NUM_WEIGHTS - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

    -- Load counter
    signal load_counter : integer range 0 to NUM_WEIGHTS := 0;
    signal load_done_reg : std_logic := '0';

    -- Save counter
    signal save_counter : integer range 0 to NUM_WEIGHTS := 0;
    signal save_done_reg : std_logic := '0';

    -- Update state
    signal update_done_reg : std_logic := '0';

    -- Helper function for weight update: w = w - lr * grad
    function apply_gradient(
        w    : std_logic_vector;
        grad : std_logic_vector;
        lr   : std_logic_vector
    ) return std_logic_vector is
        variable w_sf   : sfixed_bus;
        variable g_sf   : sfixed_bus;
        variable lr_sf  : sfixed_bus;
        variable delta  : sfixed_bus;
        variable result : sfixed_bus;
    begin
        w_sf  := to_sfixed(w, INT_BITS - 1, -FRAC_BITS);
        g_sf  := to_sfixed(grad, INT_BITS - 1, -FRAC_BITS);
        lr_sf := to_sfixed(lr, INT_BITS - 1, -FRAC_BITS);

        -- delta = lr * grad
        delta := resize(lr_sf * g_sf, INT_BITS - 1, -FRAC_BITS);

        -- result = w - delta
        result := resize(w_sf - delta, INT_BITS - 1, -FRAC_BITS);

        return to_std_logic_vector(result);
    end function;

begin

    -- Continuous output of all weights
    weights_o <= weight_regs;

    -- Output current indices
    load_idx <= load_counter when load_counter < NUM_WEIGHTS else NUM_WEIGHTS - 1;
    save_idx <= save_counter when save_counter < NUM_WEIGHTS else NUM_WEIGHTS - 1;

    -- Done signals
    load_done <= load_done_reg;
    save_done <= save_done_reg;
    update_done <= update_done_reg;

    -- Save data output
    save_data <= weight_regs(save_counter) when save_counter < NUM_WEIGHTS
                 else weight_regs(NUM_WEIGHTS - 1);

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                weight_regs <= (others => (others => '0'));
                load_counter <= 0;
                save_counter <= 0;
                load_done_reg <= '0';
                save_done_reg <= '0';
                update_done_reg <= '0';

            else
                -- Clear pulse signals
                load_done_reg <= '0';
                save_done_reg <= '0';
                update_done_reg <= '0';

                -- Load operation: store incoming weight, increment counter
                if load_en = '1' then
                    if load_counter < NUM_WEIGHTS then
                        weight_regs(load_counter) <= load_data;
                        if load_counter = NUM_WEIGHTS - 1 then
                            load_done_reg <= '1';
                            load_counter <= 0;  -- Reset for next load sequence
                        else
                            load_counter <= load_counter + 1;
                        end if;
                    end if;
                end if;

                -- Save operation: increment counter, done when complete
                if save_en = '1' then
                    if save_counter < NUM_WEIGHTS then
                        if save_counter = NUM_WEIGHTS - 1 then
                            save_done_reg <= '1';
                            save_counter <= 0;  -- Reset for next save sequence
                        else
                            save_counter <= save_counter + 1;
                        end if;
                    end if;
                end if;

                -- Update operation: apply all gradients in one cycle
                if update_en = '1' then
                    for i in 0 to NUM_WEIGHTS - 1 loop
                        weight_regs(i) <= apply_gradient(
                            weight_regs(i),
                            grad_data(i),
                            learn_rate
                        );
                    end loop;
                    update_done_reg <= '1';
                end if;

            end if;
        end if;
    end process;

end architecture rtl;

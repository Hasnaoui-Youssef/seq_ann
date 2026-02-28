library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

entity memory_control_unit is
    generic (
        ADDR_WIDTH  : integer := 10
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Host Interface
        host_write_en   : in std_logic;
        host_addr       : in integer;
        host_data_in    : in std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Calculation Unit Interface
        read_req        : in std_logic;
        read_addr       : in integer;
        read_data_out   : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        read_valid      : out std_logic;

        -- Update Interface
        update_en       : in std_logic;
        update_addr     : in integer;
        update_grad     : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        learning_rate   : in std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity memory_control_unit;

architecture rtl of memory_control_unit is

    -- BRAM Signals
    signal bram_wea : std_logic;
    signal bram_addra : std_logic_vector(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal bram_dia : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal bram_addrb : std_logic_vector(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal bram_dob : std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- Pipeline Signals for Update (Read-Modify-Write)
    signal update_en_d1   : std_logic;
    signal update_addr_d1 : integer := 0;
    signal update_grad_d1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lr_d1          : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal update_data_d1 : std_logic_vector(DATA_WIDTH - 1 downto 0); -- Latched w_old
    signal read_req_d1    : std_logic;

    -- Host write tracking for RMW race prevention
    signal host_wrote_d1     : std_logic := '0';
    signal host_write_addr_d1: integer := 0;

    -- Pending read latch for reads during updates
    signal read_pending      : std_logic := '0';
    signal read_pending_addr : integer := 0;

    function calc_update(
        w_old : std_logic_vector;
        lr    : std_logic_vector;
        grad  : std_logic_vector
    ) return std_logic_vector is
        variable w_sfixed : sfixed_bus;
        variable lr_sfixed : sfixed_bus;
        variable grad_sfixed : sfixed_bus;
        variable delta : sfixed_bus;
        variable result : sfixed_bus;
    begin
        w_sfixed := to_sfixed(w_old, w_sfixed);
        lr_sfixed := to_sfixed(lr, lr_sfixed);
        grad_sfixed := to_sfixed(grad, grad_sfixed);

        -- delta = lr * grad
        delta := resize(lr_sfixed * grad_sfixed, delta);

        -- result = w_old - delta
        result := resize(w_sfixed - delta, result);

        return to_std_logic_vector(result);
    end function;


    signal read_req_prev : std_logic := '0';
    signal read_req_d2 : std_logic := '0';
begin

    -- Port A: Writes (Host or Update)
    -- Port B: Reads (Calc or Update)
    u_bram : entity work.bram
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            ADDR_WIDTH => ADDR_WIDTH
        )
        port map (
            clka => clk,
            clkb => clk,
            ena => '1',
            enb => '1',
            wea => bram_wea,
            web => '0', -- Port B is Read Only
            addra => bram_addra,
            addrb => bram_addrb,
            dia => bram_dia,
            dib => (others => '0'),
            doa => open,
            dob => bram_dob
        );

    -- Port B Address Control (Priority: Update > Pending Read > New Read)
    process(update_en, update_addr, read_addr, read_pending, read_pending_addr)
    begin
        if update_en = '1' then
            if update_addr >= 0 and update_addr < 2**ADDR_WIDTH then
                bram_addrb <= std_logic_vector(to_unsigned(update_addr, ADDR_WIDTH));
            else
                bram_addrb <= (others => '0');
            end if;
        elsif read_pending = '1' then
            if read_pending_addr >= 0 and read_pending_addr < 2**ADDR_WIDTH then
                bram_addrb <= std_logic_vector(to_unsigned(read_pending_addr, ADDR_WIDTH));
            else
                bram_addrb <= (others => '0');
            end if;
        else
            if read_addr >= 0 and read_addr < 2**ADDR_WIDTH then
                bram_addrb <= std_logic_vector(to_unsigned(read_addr, ADDR_WIDTH));
            else
                bram_addrb <= (others => '0');
            end if;
        end if;
    end process;

    -- Port A Control (Write Priority: Host > Update)
    -- Suppress update write-back if host wrote to the same address last cycle
    process(host_write_en, host_addr, host_data_in, update_en_d1, update_addr_d1,
            update_data_d1, lr_d1, update_grad_d1, host_wrote_d1, host_write_addr_d1)
        variable suppress_update : boolean;
    begin
        -- Suppress update if host wrote to this exact address last cycle
        suppress_update := (host_wrote_d1 = '1') and (host_write_addr_d1 = update_addr_d1);

        if host_write_en = '1' then
            bram_wea <= '1';
            if host_addr >= 0 and host_addr < 2**ADDR_WIDTH then
                bram_addra <= std_logic_vector(to_unsigned(host_addr, ADDR_WIDTH));
            else
                bram_addra <= (others => '0');
            end if;
            bram_dia <= host_data_in;
        elsif update_en_d1 = '1' and not suppress_update then
            bram_wea <= '1';
            if update_addr_d1 >= 0 and update_addr_d1 < 2**ADDR_WIDTH then
                bram_addra <= std_logic_vector(to_unsigned(update_addr_d1, ADDR_WIDTH));
            else
                bram_addra <= (others => '0');
            end if;
            -- Calculate new weight using latched old weight
            bram_dia <= calc_update(update_data_d1, lr_d1, update_grad_d1);
        else
            bram_wea <= '0';
            bram_addra <= (others => '0');
            bram_dia <= (others => '0');
        end if;
    end process;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                update_en_d1 <= '0';
                read_valid <= '0';
                read_req_d1 <= '0';
                read_req_d2 <= '0';
                read_req_prev <= '0';
                host_wrote_d1 <= '0';
                host_write_addr_d1 <= 0;
                read_pending <= '0';
                read_pending_addr <= 0;
            else
                -- Track host writes for RMW race prevention
                host_wrote_d1 <= host_write_en;
                host_write_addr_d1 <= host_addr;

                -- Pipeline Update Signals (Stage 1)
                update_en_d1 <= update_en;
                update_addr_d1 <= update_addr;
                update_grad_d1 <= update_grad;
                lr_d1 <= learning_rate;

                -- Capture Read Data (for Update)
                update_data_d1 <= bram_dob;

                read_req_prev <= read_req;

                -- Read request handling with pending latch
                if read_req = '1' and read_req_prev = '0' then
                    if update_en = '0' and read_pending = '0' then
                        -- No conflict: process immediately
                        read_req_d1 <= '1';
                    else
                        -- Update in progress: latch read for later
                        read_pending <= '1';
                        read_pending_addr <= read_addr;
                        read_req_d1 <= '0';
                    end if;
                elsif read_pending = '1' and update_en = '0' then
                    -- Service the pending read now that update is done
                    read_req_d1 <= '1';
                    read_pending <= '0';
                else
                    read_req_d1 <= '0';
                end if;

                read_req_d2 <= read_req_d1;

                -- Read Valid Logic
                -- Data is valid 2 cycles after request edge
                if read_req_d2 = '1' then
                    read_valid <= '1';
                    read_data_out <= bram_dob;
                else
                    read_valid <= '0';
                end if;
            end if;
        end if;
    end process;

end architecture rtl;

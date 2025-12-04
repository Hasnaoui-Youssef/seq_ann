library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_pkg.all;
use work.types.all;
use work.pkg_layer.all;

entity memory_control_unit is
    generic (
        MEMORY_SIZE : integer := 1024;
        ADDR_WIDTH  : integer := 10
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Host Interface (Loading initial weights)
        host_write_en   : in std_logic;
        host_addr       : in integer;
        host_data_in    : in std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Calculation Unit Interface (Reading weights)
        read_req        : in std_logic;
        read_addr       : in integer;
        read_data_out   : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        read_valid      : out std_logic;

        -- Update Interface (Applying gradients)
        update_en       : in std_logic;
        update_addr     : in integer;
        update_grad     : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        learning_rate   : in std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity memory_control_unit;

architecture rtl of memory_control_unit is

    -- BRAM Component
    component bram is
        generic(
            DATA_WIDTH : integer := 16;
            ADDR_WIDTH : integer := 16
        );
        port(
            clka : in std_logic;
            clkb : in std_logic;
            ena : in std_logic;
            enb : in std_logic;
            wea : in std_logic;
            web : in std_logic;
            addra : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
            addrb : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
            dia : in std_logic_vector(DATA_WIDTH - 1 downto 0);
            dib : in std_logic_vector(DATA_WIDTH - 1 downto 0);
            doa : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            dob : out std_logic_vector(DATA_WIDTH - 1 downto 0)
        );
    end component;

    -- BRAM Signals
    signal bram_wea : std_logic;
    signal bram_addra : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    signal bram_dia : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal bram_addrb : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    signal bram_dob : std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- Pipeline Signals for Update (Read-Modify-Write)
    signal update_en_d1   : std_logic;
    signal update_addr_d1 : integer := 0;
    signal update_grad_d1 : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal lr_d1          : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal update_data_d1 : std_logic_vector(DATA_WIDTH - 1 downto 0); -- Latched w_old
    signal read_req_d1    : std_logic;

    -- Helper for fixed-point subtraction: W_new = W_old - (lr * grad)
    function calc_update(
        w_old : std_logic_vector;
        lr    : std_logic_vector;
        grad  : std_logic_vector
    ) return std_logic_vector is
        variable w_sfixed : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable lr_sfixed : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable grad_sfixed : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable delta : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
        variable result : sfixed(DATA_WIDTH/2 - 1 downto -DATA_WIDTH/2);
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

begin

    -- Instantiate BRAM
    -- Port A: Writes (Host or Update)
    -- Port B: Reads (Calc or Update)
    u_bram : bram
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

    -- Port B Address Control (Read Priority: Update > Calc)
    process(update_en, update_addr, read_addr)
    begin
        if update_en = '1' then
            if update_addr >= 0 and update_addr < 2**ADDR_WIDTH then
                bram_addrb <= std_logic_vector(to_unsigned(update_addr, ADDR_WIDTH));
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
    process(host_write_en, host_addr, host_data_in, update_en_d1, update_addr_d1, update_data_d1, lr_d1, update_grad_d1)
    begin
        if host_write_en = '1' then
            bram_wea <= '1';
            if host_addr >= 0 and host_addr < 2**ADDR_WIDTH then
                bram_addra <= std_logic_vector(to_unsigned(host_addr, ADDR_WIDTH));
            else
                bram_addra <= (others => '0');
            end if;
            bram_dia <= host_data_in;
        elsif update_en_d1 = '1' then
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

    -- Pipeline and Output Process
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                update_en_d1 <= '0';
                read_valid <= '0';
                read_req_d1 <= '0';
            else
                -- Pipeline Update Signals (Stage 1)
                update_en_d1 <= update_en;
                update_addr_d1 <= update_addr;
                update_grad_d1 <= update_grad;
                lr_d1 <= learning_rate;
                
                -- Capture Read Data (for Update)
                update_data_d1 <= bram_dob;

                -- Read Request Pipeline
                -- Only latch read request if not preempted by update
                if read_req = '1' and update_en = '0' then
                    read_req_d1 <= '1';
                else
                    read_req_d1 <= '0';
                end if;

                -- Read Valid Logic
                -- Data is valid 1 cycle after BRAM address sample (which happens when read_req_d1 is latched)
                -- So when read_req_d1 is 1, bram_dob holds the data.
                if read_req_d1 = '1' then
                    read_valid <= '1';
                    read_data_out <= bram_dob; -- Register output
                else
                    read_valid <= '0';
                end if;
            end if;
        end if;
    end process;

end architecture rtl;

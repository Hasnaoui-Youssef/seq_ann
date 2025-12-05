library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;

entity neural_network is
    generic(
        NUM_INPUTS  : integer := 2;
        NUM_LAYERS  : integer := 2;
        LAYER_SIZES : layer_config_array;  -- e.g., (3, 1) for hidden=3, output=1
        MEMORY_SIZE : integer := 1024
    );
    port (
        clk : in std_logic;
        rst : in std_logic;

        -- Control Interface
        start       : in std_logic;
        train_mode  : in std_logic;
        ready       : out std_logic;
        done        : out std_logic;

        -- Host Interface (Weight Loading)
        host_write_en : in std_logic;
        host_addr     : in integer;
        host_data     : in std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- Data Interface (Streaming)
        input_data   : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        input_valid  : in std_logic;
        input_last   : in std_logic;

        output_data  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        output_valid : out std_logic;
        output_last  : out std_logic
    );
end entity neural_network;

architecture rtl of neural_network is

    -- Internal Signals
    signal calc_mode   : std_logic;
    signal calc_start  : std_logic;
    signal calc_store  : std_logic;
    signal calc_done   : std_logic := '0';
    signal calc_ready  : std_logic;
    signal learning_rate : std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- Memory <-> Calc Interface
    signal mem_read_req   : std_logic;
    signal mem_read_addr  : integer := 0;
    signal mem_read_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal mem_read_valid : std_logic;

    signal mem_update_en   : std_logic;
    signal mem_update_addr : integer := 0;
    signal mem_update_grad : std_logic_vector(DATA_WIDTH - 1 downto 0);

begin

    -- Control Unit
    u_control : entity work.control_unit
        port map (
            clk => clk,
            rst => rst,
            start => start,
            train_mode => train_mode,
            ready => ready,
            done => done,
            calc_mode => calc_mode,
            calc_start => calc_start,
            calc_store => calc_store,
            calc_done => calc_done,
            calc_ready => calc_ready,
            learning_rate => learning_rate
        );

    -- Memory Control Unit
    u_mem : entity work.memory_control_unit
        port map (
            clk => clk,
            rst => rst,
            host_write_en => host_write_en,
            host_addr => host_addr,
            host_data_in => host_data,
            read_req => mem_read_req,
            read_addr => mem_read_addr,
            read_data_out => mem_read_data,
            read_valid => mem_read_valid,
            update_en => mem_update_en,
            update_addr => mem_update_addr,
            update_grad => mem_update_grad,
            learning_rate => learning_rate
        );

    -- Calculation Unit
    u_calc : entity work.calculation_unit
        generic map (
            NUM_INPUTS  => NUM_INPUTS,
            NUM_LAYERS  => NUM_LAYERS,
            LAYER_SIZES => LAYER_SIZES
        )
        port map (
            clk => clk,
            rst => rst,
            mode => calc_mode,
            start_store => calc_store,
            input_data => input_data,
            input_valid => input_valid,
            input_last => input_last,
            output_data => output_data,
            output_valid => output_valid,
            output_last => output_last,
            error_in => (others => '0'),
            error_in_valid => '0',
            mem_read_req => mem_read_req,
            mem_read_addr => mem_read_addr,
            mem_read_data => mem_read_data,
            mem_read_valid => mem_read_valid,
            mem_update_en => mem_update_en,
            mem_update_addr => mem_update_addr,
            mem_update_grad => mem_update_grad,
            ready => calc_ready
        );

end architecture rtl;

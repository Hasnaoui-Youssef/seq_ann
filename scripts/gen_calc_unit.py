#!/usr/bin/env python3
"""
Generate a calculation_unit.vhd for heterogeneous layer networks.

Given a network topology with mixed layer types (dense, conv2d, maxpool, etc.),
generates a custom calculation_unit with the correct generate block instantiating
each layer type with its specific parameters.

This follows the project guideline: "Hardware is generated at compile time,
not configured at runtime."
"""

import sys
from pathlib import Path
from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from onnx_parser import (
    ParsedNetwork,
    DenseLayerInfo,
    ConvLayerInfo,
    PoolLayerInfo,
    FlattenLayerInfo,
    RNNLayerInfo,
    LayerInfo,
)


def compute_layer_output_size(layer: LayerInfo) -> int:
    """Get the flat output size of a layer."""
    return layer.num_outputs


def compute_weight_count(layer: LayerInfo) -> int:
    """Get the number of weights for a layer."""
    if isinstance(layer, DenseLayerInfo):
        return (layer.num_inputs + 1) * layer.num_outputs
    elif isinstance(layer, ConvLayerInfo):
        return (layer.kernel_h * layer.kernel_w * layer.c_in + 1) * layer.num_filters
    elif isinstance(layer, RNNLayerInfo):
        if layer.layer_type == "lstm":
            return 4 * (layer.input_size + layer.hidden_size + 1) * layer.hidden_size
        else:
            return (layer.input_size + layer.hidden_size + 1) * layer.hidden_size
    else:
        return 0  # Pool, Flatten have no weights


def compute_layer_input_size(network: ParsedNetwork, layer_idx: int) -> int:
    """Get the flat input size for a layer."""
    if layer_idx == 0:
        return network.num_inputs
    return compute_layer_output_size(network.layers[layer_idx - 1])


def generate_layer_instantiation(layer: LayerInfo, idx: int, input_size: int) -> str:
    """Generate VHDL instantiation code for a specific layer."""
    output_size = compute_layer_output_size(layer)
    weight_count = compute_weight_count(layer)

    if isinstance(layer, DenseLayerInfo):
        use_sigmoid = "true" if layer.activation == "sigmoid" else "false"
        return f"""    -- Layer {idx}: Dense ({input_size} -> {output_size}, {layer.activation})
    gen_layer_{idx}: block
        constant THIS_INPUT_SIZE  : integer := {input_size};
        constant THIS_OUTPUT_SIZE : integer := {output_size};
        constant THIS_WEIGHT_COUNT: integer := {weight_count};
        signal layer_grads : std_logic_bus_array(0 to THIS_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    begin
        u_layer_{idx}: entity work.layer
            generic map (
                NUM_INPUTS => THIS_INPUT_SIZE,
                LAYER_SIZE => THIS_OUTPUT_SIZE,
                USE_SIGMOID => {use_sigmoid}
            )
            port map (
                clk => clk, rst => rst,
                fwd_en => not in_backward_phase,
                bwd_en => in_backward_phase,
                fwd_ctrl_in => fwd_ctrl({idx}),
                fwd_data_in => fwd_data({idx})(0 to THIS_INPUT_SIZE - 1),
                fwd_ctrl_out => fwd_ctrl({idx + 1}),
                fwd_data_out => fwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_in => bwd_ctrl({idx + 1}),
                bwd_error_in => bwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_out => bwd_ctrl({idx}),
                bwd_error_out => bwd_data({idx})(0 to THIS_INPUT_SIZE - 1),
                weight_load_en => layer_weight_load_en({idx}),
                weight_load_data => mem_read_data,
                weight_load_done => layer_weight_load_done({idx}),
                weight_save_en => '0',
                weight_save_data => open,
                weight_save_done => open,
                weight_update_en => layer_weight_update_en({idx}),
                weight_learn_rate => learning_rate,
                weight_update_done => layer_weight_update_done({idx}),
                grads_out => layer_grads
            );
    end block gen_layer_{idx};
"""

    elif isinstance(layer, ConvLayerInfo):
        return f"""    -- Layer {idx}: Conv2D ({layer.c_in}x{layer.h_in}x{layer.w_in} -> {layer.num_filters}x{layer.h_out}x{layer.w_out})
    gen_layer_{idx}: block
        constant THIS_INPUT_SIZE  : integer := {input_size};
        constant THIS_OUTPUT_SIZE : integer := {output_size};
        constant THIS_WEIGHT_COUNT: integer := {weight_count};
        signal layer_grads : std_logic_bus_array(0 to THIS_WEIGHT_COUNT - 1)(DATA_WIDTH - 1 downto 0);
    begin
        u_layer_{idx}: entity work.conv2d_layer
            generic map (
                C_IN => {layer.c_in}, H_IN => {layer.h_in}, W_IN => {layer.w_in},
                NUM_FILTERS => {layer.num_filters},
                KERNEL_H => {layer.kernel_h}, KERNEL_W => {layer.kernel_w},
                STRIDE_H => {layer.stride_h}, STRIDE_W => {layer.stride_w},
                PAD_H => {layer.pad_h}, PAD_W => {layer.pad_w}
            )
            port map (
                clk => clk, rst => rst,
                fwd_ctrl_in => fwd_ctrl({idx}),
                fwd_data_in => fwd_data({idx})(0 to THIS_INPUT_SIZE - 1),
                fwd_ctrl_out => fwd_ctrl({idx + 1}),
                fwd_data_out => fwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_in => bwd_ctrl({idx + 1}),
                bwd_error_in => bwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_out => bwd_ctrl({idx}),
                bwd_error_out => bwd_data({idx})(0 to THIS_INPUT_SIZE - 1),
                weight_load_en => layer_weight_load_en({idx}),
                weight_load_data => mem_read_data,
                weight_load_done => layer_weight_load_done({idx}),
                weight_save_en => '0',
                weight_save_data => open,
                weight_save_done => open,
                weight_update_en => layer_weight_update_en({idx}),
                weight_learn_rate => learning_rate,
                weight_update_done => layer_weight_update_done({idx}),
                grads_out => layer_grads
            );
    end block gen_layer_{idx};
"""

    elif isinstance(layer, PoolLayerInfo):
        entity = "maxpool_layer" if layer.layer_type == "maxpool" else "avgpool_layer"
        return f"""    -- Layer {idx}: {layer.layer_type.title()} ({layer.c_in}x{layer.h_in}x{layer.w_in} -> {layer.c_in}x{layer.h_out}x{layer.w_out})
    gen_layer_{idx}: block
        constant THIS_INPUT_SIZE  : integer := {input_size};
        constant THIS_OUTPUT_SIZE : integer := {output_size};
    begin
        u_layer_{idx}: entity work.{entity}
            generic map (
                CHANNELS => {layer.c_in}, H_IN => {layer.h_in}, W_IN => {layer.w_in},
                POOL_H => {layer.kernel_h}, POOL_W => {layer.kernel_w},
                STRIDE_H => {layer.stride_h}, STRIDE_W => {layer.stride_w}
            )
            port map (
                clk => clk, rst => rst,
                fwd_ctrl_in => fwd_ctrl({idx}),
                fwd_data_in => fwd_data({idx})(0 to THIS_INPUT_SIZE - 1),
                fwd_ctrl_out => fwd_ctrl({idx + 1}),
                fwd_data_out => fwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_in => bwd_ctrl({idx + 1}),
                bwd_error_in => bwd_data({idx + 1})(0 to THIS_OUTPUT_SIZE - 1),
                bwd_ctrl_out => bwd_ctrl({idx}),
                bwd_error_out => bwd_data({idx})(0 to THIS_INPUT_SIZE - 1)
            );
        -- No weights for pooling layers
        layer_weight_load_done({idx}) <= '1';
        layer_weight_update_done({idx}) <= '1';
    end block gen_layer_{idx};
"""

    elif isinstance(layer, FlattenLayerInfo):
        return f"""    -- Layer {idx}: Flatten ({input_size} -> {output_size})
    gen_layer_{idx}: block
        constant THIS_SIZE : integer := {input_size};
    begin
        u_layer_{idx}: entity work.flatten_layer
            generic map (TOTAL_SIZE => THIS_SIZE)
            port map (
                clk => clk, rst => rst,
                fwd_ctrl_in => fwd_ctrl({idx}),
                fwd_data_in => fwd_data({idx})(0 to THIS_SIZE - 1),
                fwd_ctrl_out => fwd_ctrl({idx + 1}),
                fwd_data_out => fwd_data({idx + 1})(0 to THIS_SIZE - 1),
                bwd_ctrl_in => bwd_ctrl({idx + 1}),
                bwd_error_in => bwd_data({idx + 1})(0 to THIS_SIZE - 1),
                bwd_ctrl_out => bwd_ctrl({idx}),
                bwd_error_out => bwd_data({idx})(0 to THIS_SIZE - 1)
            );
        -- No weights for flatten
        layer_weight_load_done({idx}) <= '1';
        layer_weight_update_done({idx}) <= '1';
    end block gen_layer_{idx};
"""

    else:
        return f"    -- Layer {idx}: {type(layer).__name__} (not yet supported in generation)\n"


def generate_calculation_unit(network: ParsedNetwork, output_path: str) -> None:
    """Generate a complete calculation_unit.vhd for the given network."""
    num_layers = len(network.layers)
    num_inputs = network.num_inputs
    num_outputs = compute_layer_output_size(network.layers[-1])

    # Compute max size across all layers
    max_size = num_inputs
    for layer in network.layers:
        s = compute_layer_output_size(layer)
        if s > max_size:
            max_size = s

    # Compute total weights
    total_weights = sum(compute_weight_count(layer) for layer in network.layers)

    # Compute weight count per layer for FSM
    layer_weight_counts = []
    for layer in network.layers:
        layer_weight_counts.append(compute_weight_count(layer))

    # Generate layer weight count VHDL constant array
    wc_list = ", ".join(str(wc) for wc in layer_weight_counts)

    # Generate layer instantiations
    layer_blocks = []
    for i, layer in enumerate(network.layers):
        in_size = compute_layer_input_size(network, i)
        layer_blocks.append(generate_layer_instantiation(layer, i, in_size))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vhdl = f"""\
-- Auto-generated calculation_unit for heterogeneous network
-- Network: {network.name}
-- Layers: {num_layers}
-- Total weights: {total_weights}

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;
use work.pkg_layer.all;
use IEEE.fixed_pkg.all;

entity calculation_unit is
    generic (
        NUM_INPUTS  : integer := {num_inputs};
        NUM_LAYERS  : integer := {num_layers};
        LAYER_SIZES : layer_config_array := ({", ".join(str(compute_layer_output_size(l)) for l in network.layers)});
        USE_SIGMOID : boolean_array := ({", ".join("true" if (isinstance(l, DenseLayerInfo) and l.activation == "sigmoid") else "false" for l in network.layers)})
    );
    port (
        clk : in std_logic;
        rst : in std_logic;
        load_weights : in std_logic;
        start        : in std_logic;
        mode         : in std_logic;
        start_store  : in std_logic;
        weights_loaded : out std_logic;
        ready          : out std_logic;
        done           : out std_logic;
        output_data  : out std_logic_bus_array(0 to {num_outputs} - 1)(DATA_WIDTH - 1 downto 0);
        output_valid : out std_logic;
        learning_rate : in std_logic_vector(DATA_WIDTH - 1 downto 0);
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

    function compute_error(
        output_val : std_logic_vector;
        target_val : std_logic_vector
    ) return std_logic_vector is
        variable out_sf : sfixed_bus;
        variable tgt_sf : sfixed_bus;
        variable err_sf : sfixed_bus;
    begin
        out_sf := to_sfixed(output_val, INT_BITS - 1, -FRAC_BITS);
        tgt_sf := to_sfixed(target_val, INT_BITS - 1, -FRAC_BITS);
        err_sf := resize(out_sf - tgt_sf, INT_BITS - 1, -FRAC_BITS);
        return to_std_logic_vector(err_sf);
    end function;

    constant MAX_SIZE      : integer := {max_size};
    constant NUM_OUTPUTS   : integer := {num_outputs};
    constant TOTAL_WEIGHTS : integer := {total_weights};
    constant TARGET_BASE_ADDR : integer := {num_inputs};
    constant WEIGHT_BASE_ADDR : integer := {num_inputs} + {num_outputs};

    type weight_count_array is array(0 to {num_layers} - 1) of integer;
    constant LAYER_WEIGHT_COUNTS : weight_count_array := ({wc_list});

    type ctrl_array_t is array (0 to {num_layers}) of layer_control_t;
    type data_array_t is array (0 to {num_layers}) of std_logic_bus_array(0 to MAX_SIZE - 1)(DATA_WIDTH - 1 downto 0);

    constant CTRL_INIT : layer_control_t := (valid => '0', last => '0');
    constant DATA_INIT : std_logic_bus_array(0 to MAX_SIZE - 1)(DATA_WIDTH - 1 downto 0) := (others => (others => '0'));

    signal fwd_ctrl : ctrl_array_t := (others => CTRL_INIT);
    signal fwd_data : data_array_t := (others => DATA_INIT);
    signal bwd_ctrl : ctrl_array_t := (others => CTRL_INIT);
    signal bwd_data : data_array_t := (others => DATA_INIT);

    signal input_buffer : std_logic_bus_array(0 to {num_inputs} - 1)(DATA_WIDTH - 1 downto 0);

    signal layer_weight_load_en   : std_logic_vector(0 to {num_layers} - 1) := (others => '0');
    signal layer_weight_load_done : std_logic_vector(0 to {num_layers} - 1);
    signal layer_weight_update_en   : std_logic_vector(0 to {num_layers} - 1) := (others => '0');
    signal layer_weight_update_done : std_logic_vector(0 to {num_layers} - 1);

    signal mode_latched : std_logic := '0';
    signal in_backward_phase : std_logic := '0';

    type state_t is (
        IDLE, LOAD_WEIGHTS_REQ, LOAD_WEIGHTS_WAIT, WEIGHTS_READY,
        FETCH_INPUTS_REQ, FETCH_INPUTS_WAIT,
        FORWARD_START, FORWARD_WAIT,
        LOAD_TARGET_REQ, LOAD_TARGET_WAIT,
        BACKWARD_START, BACKWARD_WAIT,
        UPDATE_WEIGHTS, UPDATE_WEIGHTS_WAIT,
        DONE_STATE
    );
    signal state : state_t := IDLE;

    signal fetch_idx : integer := 0;
    signal current_layer : integer range 0 to {num_layers} - 1 := 0;
    signal layer_weight_idx : integer := 0;
    signal weights_loaded_reg : std_logic := '0';
    signal fwd_complete : std_logic := '0';
    signal bwd_complete : std_logic := '0';

    signal output_data_reg : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));
    signal output_valid_reg : std_logic := '0';
    signal target_buffer : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0)
        := (others => (others => '0'));

begin

    fwd_complete <= fwd_ctrl({num_layers}).valid;
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
                bwd_ctrl({num_layers}).valid <= '0';
                bwd_ctrl({num_layers}).last <= '0';
                bwd_data({num_layers}) <= (others => (others => '0'));
                input_buffer <= (others => (others => '0'));
                target_buffer <= (others => (others => '0'));
                output_data_reg <= (others => (others => '0'));
                output_valid_reg <= '0';
                mode_latched <= '0';
                in_backward_phase <= '0';
            else
                mem_read_req <= '0';
                mem_update_en <= '0';
                fwd_ctrl(0).valid <= '0';
                fwd_ctrl(0).last <= '0';
                bwd_ctrl({num_layers}).valid <= '0';
                bwd_ctrl({num_layers}).last <= '0';

                case state is
                    when IDLE =>
                        ready <= '0';
                        done <= '0';
                        if load_weights = '1' then
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
                        if LAYER_WEIGHT_COUNTS(current_layer) = 0 then
                            -- Skip layers with no weights (pool, flatten)
                            if current_layer = {num_layers} - 1 then
                                weights_loaded_reg <= '1';
                                state <= WEIGHTS_READY;
                            else
                                current_layer <= current_layer + 1;
                                state <= LOAD_WEIGHTS_REQ;
                            end if;
                        else
                            mem_read_addr <= WEIGHT_BASE_ADDR + fetch_idx;
                            mem_read_req <= '1';
                            state <= LOAD_WEIGHTS_WAIT;
                        end if;

                    when LOAD_WEIGHTS_WAIT =>
                        if mem_read_valid = '1' then
                            if layer_weight_idx = LAYER_WEIGHT_COUNTS(current_layer) - 1 then
                                layer_weight_idx <= 0;
                                if current_layer = {num_layers} - 1 then
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
                            input_buffer(fetch_idx) <= mem_read_data;
                            if fetch_idx = {num_inputs} - 1 then
                                state <= FORWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= FETCH_INPUTS_REQ;
                            end if;
                        end if;

                    when FORWARD_START =>
                        fwd_data(0)(0 to {num_inputs} - 1) <= input_buffer;
                        fwd_ctrl(0).valid <= '1';
                        fwd_ctrl(0).last <= '1';
                        state <= FORWARD_WAIT;

                    when FORWARD_WAIT =>
                        if fwd_complete = '1' then
                            for i in 0 to NUM_OUTPUTS - 1 loop
                                output_data_reg(i) <= fwd_data({num_layers})(i);
                            end loop;
                            output_valid_reg <= '1';
                            if mode_latched = '1' then
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
                            target_buffer(fetch_idx) <= mem_read_data;
                            if fetch_idx = NUM_OUTPUTS - 1 then
                                state <= BACKWARD_START;
                            else
                                fetch_idx <= fetch_idx + 1;
                                state <= LOAD_TARGET_REQ;
                            end if;
                        end if;

                    when BACKWARD_START =>
                        in_backward_phase <= '1';
                        for i in 0 to NUM_OUTPUTS - 1 loop
                            bwd_data({num_layers})(i) <= compute_error(
                                output_data_reg(i), target_buffer(i));
                        end loop;
                        bwd_ctrl({num_layers}).valid <= '1';
                        bwd_ctrl({num_layers}).last <= '1';
                        state <= BACKWARD_WAIT;

                    when BACKWARD_WAIT =>
                        bwd_ctrl({num_layers}).valid <= '0';
                        bwd_ctrl({num_layers}).last <= '0';
                        if bwd_complete = '1' then
                            state <= UPDATE_WEIGHTS;
                        end if;

                    when UPDATE_WEIGHTS =>
                        layer_weight_update_en <= (others => '1');
                        state <= UPDATE_WEIGHTS_WAIT;

                    when UPDATE_WEIGHTS_WAIT =>
                        layer_weight_update_en <= (others => '0');
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

    -- Layer instantiations (generated per network topology)
{"".join(layer_blocks)}
    output_valid <= output_valid_reg;
    output_data  <= output_data_reg;

end architecture rtl;
"""

    output_path.write_text(vhdl)
    print(f"  Generated {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate heterogeneous calculation_unit")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output", default=str(ROOT_DIR / "src" / "units" / "calculation_unit.vhd"))
    args = parser.parse_args()

    from onnx_parser import parse_onnx, print_network_summary

    network = parse_onnx(args.model)
    print_network_summary(network)
    generate_calculation_unit(network, args.output)
    print("Done!")

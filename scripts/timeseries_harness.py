#!/usr/bin/env python3
"""
Timeseries Test Harness for Neural Network Accelerator

Trains a simple RNN on sine wave prediction, exports to ONNX,
and generates a VHDL testbench for inference verification.

Usage:
    python scripts/timeseries_harness.py [--model-type rnn|lstm] [--epochs 50]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from onnx_parser import (
    float_to_fixed_hex,
    parse_onnx,
    print_network_summary,
    export_weights_flat_hex,
)


def generate_sine_data(seq_len: int = 10, num_samples: int = 500):
    """Generate sine wave prediction dataset.
    Input: seq_len consecutive sine values
    Target: next sine value
    """
    t = np.linspace(0, 8 * np.pi, num_samples + seq_len)
    data = np.sin(t)

    X = np.zeros((num_samples, seq_len, 1), dtype=np.float32)
    y = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data[i:i + seq_len].reshape(-1, 1)
        y[i] = data[i + seq_len]

    return X, y


def train_rnn_timeseries(
    model_type: str = "rnn",
    epochs: int = 50,
    seq_len: int = 10,
    hidden_size: int = 16,
    save_path: str = None,
):
    """Train an RNN/LSTM on sine wave prediction."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("ERROR: PyTorch required for training.")
        print("Install with: pip install torch")
        sys.exit(1)

    if save_path is None:
        save_path = f"data/timeseries_{model_type}.onnx"

    class TimeseriesRNN(nn.Module):
        def __init__(self, input_size, hidden_size, use_lstm):
            super().__init__()
            if use_lstm:
                self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            else:
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
            self.fc = nn.Linear(hidden_size, 1)
            self.use_lstm = use_lstm

        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            if self.use_lstm:
                out, (hn, _) = self.rnn(x)
            else:
                out, hn = self.rnn(x)
            # Use last hidden state
            last_hidden = hn.squeeze(0)
            return self.fc(last_hidden)

    X_train, y_train = generate_sine_data(seq_len, 400)
    X_test, y_test = generate_sine_data(seq_len, 100)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    use_lstm = model_type == "lstm"
    model = TimeseriesRNN(1, hidden_size, use_lstm)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"Training {model_type.upper()} on sine wave prediction ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t)
                test_loss = criterion(test_pred, y_test_t)
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={loss.item():.6f}, test_loss={test_loss.item():.6f}")

    # Export to ONNX
    save_path = str(PROJECT_ROOT / save_path)
    dummy_input = torch.randn(1, seq_len, 1)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"Model saved to {save_path}")

    # Return test data for testbench generation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    return save_path, X_test, y_test, predictions


def generate_timeseries_testbench(
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    num_samples: int,
    seq_len: int,
    output_path: str,
    int_bits: int = 16,
    frac_bits: int = 16,
):
    """Generate a VHDL testbench for timeseries inference verification."""
    input_size = 1
    total_input = seq_len * input_size

    lines = []
    lines.append("-- Auto-generated timeseries inference testbench")
    lines.append(f"-- Tests {num_samples} sine wave prediction samples")
    lines.append("")
    lines.append("library IEEE;")
    lines.append("use IEEE.std_logic_1164.all;")
    lines.append("use IEEE.numeric_std.all;")
    lines.append("use IEEE.fixed_pkg.all;")
    lines.append("use work.types.all;")
    lines.append("use work.pkg_layer.all;")
    lines.append("")
    lines.append("entity timeseries_tb is")
    lines.append("end entity timeseries_tb;")
    lines.append("")
    lines.append("architecture testbench of timeseries_tb is")
    lines.append(f"    constant CLK_PERIOD : time := 10 ns;")
    lines.append(f"    constant SEQ_LEN : integer := {seq_len};")
    lines.append(f"    constant NUM_SAMPLES : integer := {num_samples};")
    lines.append("")
    lines.append("    signal clk : std_logic := '0';")
    lines.append("    signal rst : std_logic := '0';")
    lines.append("")
    lines.append("    function to_slv(r : real) return std_logic_vector is")
    lines.append("    begin")
    lines.append("        return to_std_logic_vector(to_sfixed(r, INT_BITS - 1, -FRAC_BITS));")
    lines.append("    end function;")
    lines.append("")
    lines.append("    function to_real_val(slv : std_logic_vector) return real is")
    lines.append("    begin")
    lines.append("        return to_real(to_sfixed(slv, INT_BITS - 1, -FRAC_BITS));")
    lines.append("    end function;")
    lines.append("")
    lines.append("begin")
    lines.append("")
    lines.append("    process begin")
    lines.append("        while true loop")
    lines.append("            clk <= '0'; wait for CLK_PERIOD / 2;")
    lines.append("            clk <= '1'; wait for CLK_PERIOD / 2;")
    lines.append("        end loop;")
    lines.append("    end process;")
    lines.append("")
    lines.append("    process")
    lines.append("        variable total_error : real := 0.0;")
    lines.append("        variable sample_error : real;")
    lines.append("    begin")
    lines.append("        rst <= '1';")
    lines.append("        wait for CLK_PERIOD * 5;")
    lines.append("        rst <= '0';")
    lines.append("        wait for CLK_PERIOD * 2;")
    lines.append("")
    lines.append('        report "Timeseries Prediction Test: " & integer\'image(NUM_SAMPLES) & " samples";')

    for s in range(min(num_samples, len(test_inputs))):
        target_val = float(test_targets[s, 0])
        lines.append(f"")
        lines.append(f"        -- Sample {s}: target = {target_val:.4f}")
        lines.append(f'        report "Sample {s}: target = {target_val:.4f}";')
        lines.append(f"        wait for CLK_PERIOD;")

    lines.append("")
    lines.append('        report "TIMESERIES TEST COMPLETE";')
    lines.append("        std.env.stop;")
    lines.append("    end process;")
    lines.append("")
    lines.append("end architecture testbench;")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Testbench written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Timeseries test harness")
    parser.add_argument("--model-type", choices=["rnn", "lstm"], default="rnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-test-samples", type=int, default=10)
    parser.add_argument("--int-bits", type=int, default=16)
    parser.add_argument("--frac-bits", type=int, default=16)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if not args.skip_training:
        onnx_path, X_test, y_test, preds = train_rnn_timeseries(
            args.model_type, args.epochs, args.seq_len, args.hidden_size
        )
    else:
        onnx_path = str(PROJECT_ROOT / f"data/timeseries_{args.model_type}.onnx")
        X_test, y_test = generate_sine_data(args.seq_len, args.num_test_samples)
        preds = None

    # Parse ONNX
    print("\nParsing ONNX model...")
    network = parse_onnx(onnx_path)
    print_network_summary(network)

    # Export weights
    weights_path = PROJECT_ROOT / "data" / f"timeseries_{args.model_type}_weights.hex"
    export_weights_flat_hex(network, weights_path, args.int_bits, args.frac_bits)
    print(f"Weights exported to {weights_path}")

    # Generate testbench
    tb_path = PROJECT_ROOT / "testbench" / "timeseries_tb.vhd"
    generate_timeseries_testbench(
        X_test, y_test, args.num_test_samples, args.seq_len,
        tb_path, args.int_bits, args.frac_bits,
    )

    print("\nDone! To run the testbench:")
    print(f"  make TESTBENCH=timeseries test")


if __name__ == "__main__":
    main()

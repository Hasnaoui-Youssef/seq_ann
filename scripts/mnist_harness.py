#!/usr/bin/env python3
"""
MNIST Test Harness for Neural Network Accelerator

Trains a small network on MNIST, exports to ONNX, generates
weight hex files and a VHDL testbench for inference verification.

Usage:
    python scripts/mnist_harness.py [--model-type dense|cnn] [--epochs 5]

Two model types:
  - dense: Simple MLP (784 -> 128 -> 64 -> 10)
  - cnn: LeNet-style CNN (Conv->Pool->Conv->Pool->Dense->Dense)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from onnx_parser import (
    ParsedNetwork,
    export_weights_flat_hex,
    float_to_fixed_hex,
    parse_onnx,
    print_network_summary,
)


def train_dense_mnist(epochs: int = 5, save_path: str = "data/mnist_dense.onnx"):
    """Train a simple dense network on MNIST and export to ONNX."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
    except ImportError:
        print("ERROR: PyTorch and torchvision required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    class DenseMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = self.fc3(x)
            return x

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading/loading MNIST dataset...")
    train_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    model = DenseMNIST()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training dense MNIST model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, acc={acc:.1f}%")

    # Export to ONNX
    save_path = str(PROJECT_ROOT / save_path)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"Model saved to {save_path}")
    return save_path, model, test_set


def train_cnn_mnist(epochs: int = 5, save_path: str = "data/mnist_cnn.onnx"):
    """Train a LeNet-style CNN on MNIST and export to ONNX."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
    except ImportError:
        print("ERROR: PyTorch and torchvision required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    class LeNetMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv: 1x28x28 -> 4x26x26
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
            # MaxPool: 4x26x26 -> 4x13x13
            self.pool1 = nn.MaxPool2d(2, 2)
            # Conv: 4x13x13 -> 8x11x11
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
            # MaxPool: 8x11x11 -> 8x5x5
            self.pool2 = nn.MaxPool2d(2, 2)
            # Dense: 200 -> 32 -> 10
            self.fc1 = nn.Linear(8 * 5 * 5, 32)
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 8 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading/loading MNIST dataset...")
    train_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    model = LeNetMNIST()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training CNN MNIST model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, acc={acc:.1f}%")

    save_path = str(PROJECT_ROOT / save_path)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"Model saved to {save_path}")
    return save_path, model, test_set


def generate_inference_testbench(
    network: ParsedNetwork,
    test_samples: list[tuple[np.ndarray, int]],
    output_path: str,
    int_bits: int = 16,
    frac_bits: int = 16,
):
    """
    Generate a VHDL testbench that loads weights and runs inference
    on a set of test samples, comparing against expected outputs.
    """
    num_inputs = network.num_inputs
    num_outputs = network.num_outputs

    # Count total weights for memory layout
    total_weights = 0
    for layer in network.layers:
        if hasattr(layer, 'weights') and hasattr(layer.weights, 'shape') and layer.weights.size > 0:
            if hasattr(layer, 'num_inputs'):
                total_weights += layer.num_inputs * layer.num_outputs + layer.num_outputs
            elif hasattr(layer, 'num_filters'):
                ksize = layer.kernel_h * layer.kernel_w * layer.c_in
                total_weights += (ksize + 1) * layer.num_filters

    num_samples = len(test_samples)

    lines = []
    lines.append("-- Auto-generated MNIST inference testbench")
    lines.append("-- Tests " + str(num_samples) + " samples against pre-trained weights")
    lines.append("")
    lines.append("library IEEE;")
    lines.append("use IEEE.std_logic_1164.all;")
    lines.append("use IEEE.numeric_std.all;")
    lines.append("use IEEE.fixed_pkg.all;")
    lines.append("use work.types.all;")
    lines.append("use work.pkg_layer.all;")
    lines.append("")
    lines.append("entity mnist_inference_tb is")
    lines.append("end entity mnist_inference_tb;")
    lines.append("")
    lines.append("architecture testbench of mnist_inference_tb is")
    lines.append(f"    constant CLK_PERIOD : time := 10 ns;")
    lines.append(f"    constant NUM_INPUTS : integer := {num_inputs};")
    lines.append(f"    constant NUM_OUTPUTS : integer := {num_outputs};")
    lines.append(f"    constant NUM_SAMPLES : integer := {num_samples};")
    lines.append("")
    lines.append("    signal clk : std_logic := '0';")
    lines.append("    signal rst : std_logic := '0';")
    lines.append("    signal start : std_logic := '0';")
    lines.append("    signal mode : std_logic := '0';")
    lines.append("    signal done : std_logic;")
    lines.append("")
    lines.append("    -- Memory interface")
    lines.append("    signal mem_write_en : std_logic := '0';")
    lines.append("    signal mem_write_addr : std_logic_vector(15 downto 0) := (others => '0');")
    lines.append("    signal mem_write_data : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');")
    lines.append("")
    lines.append("    -- Output")
    lines.append("    signal output_data : std_logic_bus_array(0 to NUM_OUTPUTS - 1)(DATA_WIDTH - 1 downto 0);")
    lines.append("    signal output_valid : std_logic;")
    lines.append("")
    lines.append("    -- Test data")
    lines.append(f"    type sample_array_t is array(0 to NUM_SAMPLES - 1) of integer;")
    lines.append(f"    constant EXPECTED_LABELS : sample_array_t := (")

    # Write expected labels
    label_strs = [str(label) for _, label in test_samples]
    lines.append("        " + ", ".join(label_strs))
    lines.append("    );")
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
    lines.append("    -- Stimulus process")
    lines.append("    process")
    lines.append("        variable correct_count : integer := 0;")
    lines.append("        variable max_val : real;")
    lines.append("        variable max_idx : integer;")
    lines.append("        variable val : real;")
    lines.append("    begin")
    lines.append("        rst <= '1';")
    lines.append("        wait for CLK_PERIOD * 5;")
    lines.append("        rst <= '0';")
    lines.append("        wait for CLK_PERIOD * 2;")
    lines.append("")
    lines.append('        report "MNIST Inference Test: " & integer\'image(NUM_SAMPLES) & " samples";')
    lines.append("")

    # For each test sample, generate the stimulus
    for s_idx, (sample_data, expected_label) in enumerate(test_samples):
        flat = sample_data.flatten()
        lines.append(f"        -- Sample {s_idx}: expected label = {expected_label}")

        # Write input data to memory
        for p_idx in range(min(len(flat), num_inputs)):
            hex_val = float_to_fixed_hex(float(flat[p_idx]), int_bits, frac_bits)
            lines.append(f'        mem_write_addr <= x"{p_idx:04X}";')
            lines.append(f'        mem_write_data <= x"{hex_val}";')
            lines.append(f'        mem_write_en <= \'1\';')
            lines.append(f"        wait until rising_edge(clk);")

        lines.append("        mem_write_en <= '0';")
        lines.append("        wait for CLK_PERIOD * 2;")
        lines.append("")
        lines.append("        -- Run inference")
        lines.append("        mode <= '0';  -- inference mode")
        lines.append("        start <= '1';")
        lines.append("        wait until rising_edge(clk);")
        lines.append("        start <= '0';")
        lines.append("        wait until done = '1';")
        lines.append("        wait until rising_edge(clk);")
        lines.append("")
        lines.append("        -- Find argmax of output")
        lines.append(f"        max_val := to_real_val(output_data(0));")
        lines.append(f"        max_idx := 0;")
        lines.append(f"        for i in 1 to NUM_OUTPUTS - 1 loop")
        lines.append(f"            val := to_real_val(output_data(i));")
        lines.append(f"            if val > max_val then")
        lines.append(f"                max_val := val;")
        lines.append(f"                max_idx := i;")
        lines.append(f"            end if;")
        lines.append(f"        end loop;")
        lines.append("")
        lines.append(f'        report "Sample {s_idx}: predicted=" & integer\'image(max_idx) '
                      f'& " expected=" & integer\'image(EXPECTED_LABELS({s_idx}));')
        lines.append(f"        if max_idx = EXPECTED_LABELS({s_idx}) then")
        lines.append(f"            correct_count := correct_count + 1;")
        lines.append(f"        end if;")
        lines.append("")

    lines.append(f'        report "Accuracy: " & integer\'image(correct_count) & "/{num_samples} = " '
                  f'& integer\'image(correct_count * 100 / {num_samples}) & "%";')
    lines.append("")
    lines.append('        report "MNIST INFERENCE TEST COMPLETE";')
    lines.append("        std.env.stop;")
    lines.append("    end process;")
    lines.append("")
    lines.append("end architecture testbench;")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Testbench written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST test harness for NN accelerator")
    parser.add_argument("--model-type", choices=["dense", "cnn"], default="dense",
                        help="Model architecture (default: dense)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--num-test-samples", type=int, default=10,
                        help="Number of test samples for VHDL testbench (default: 10)")
    parser.add_argument("--int-bits", type=int, default=16)
    parser.add_argument("--frac-bits", type=int, default=16)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing ONNX model")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="Path to existing ONNX model (with --skip-training)")
    args = parser.parse_args()

    if args.skip_training:
        if not args.onnx_path:
            args.onnx_path = f"data/mnist_{args.model_type}.onnx"
        onnx_path = str(PROJECT_ROOT / args.onnx_path)
    else:
        if args.model_type == "cnn":
            onnx_path, _, _ = train_cnn_mnist(args.epochs)
        else:
            onnx_path, _, _ = train_dense_mnist(args.epochs)

    # Parse ONNX model
    print("\nParsing ONNX model...")
    network = parse_onnx(onnx_path)
    print_network_summary(network)

    # Export weights
    weights_path = PROJECT_ROOT / "data" / f"mnist_{args.model_type}_weights.hex"
    export_weights_flat_hex(network, weights_path, args.int_bits, args.frac_bits)
    print(f"Weights exported to {weights_path}")

    # Generate test samples
    print(f"\nGenerating testbench with {args.num_test_samples} samples...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform)

        test_samples = []
        # Pick diverse samples (one per class if possible)
        by_class = {}
        for i in range(len(test_set)):
            img, label = test_set[i]
            if label not in by_class:
                by_class[label] = (img.numpy(), label)
            if len(by_class) >= min(10, args.num_test_samples):
                break

        test_samples = list(by_class.values())[:args.num_test_samples]
        # Fill remaining with random samples if needed
        idx = 0
        while len(test_samples) < args.num_test_samples and idx < len(test_set):
            img, label = test_set[idx]
            test_samples.append((img.numpy(), label))
            idx += 1

    except ImportError:
        print("WARNING: torchvision not available, generating random test samples")
        test_samples = [(np.random.randn(1, 28, 28).astype(np.float32), i % 10)
                        for i in range(args.num_test_samples)]

    tb_path = PROJECT_ROOT / "testbench" / "mnist_inference_tb.vhd"
    generate_inference_testbench(network, test_samples, tb_path, args.int_bits, args.frac_bits)

    print("\nDone! To run the testbench:")
    print(f"  make TESTBENCH=mnist_inference test")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import sys

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from config import auto_load_config, SigmoidConfig as YamlSigmoidConfig


def sigmoid(x: float) -> float:
    """Compute sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def generate_sigmoid_lut(config: YamlSigmoidConfig, frac_bits: int):
    """Generate LUT values for sigmoid function."""
    lut = []
    x_min, x_max = config.input_range
    lut_size = config.lut_size
    
    for i in range(lut_size):
        x = x_min + (x_max - x_min) * i / (lut_size - 1)
        sig_val = sigmoid(x)
        lut.append((i, x, sig_val))
    
    # Force edge values to exact 0 and 1 for proper saturation behavior
    idx_first, x_first, _ = lut[0]
    idx_last, x_last, _ = lut[-1]
    lut[0] = (idx_first, x_first, 0.0)
    lut[-1] = (idx_last, x_last, 1.0)

    return lut


def float_to_fixed_bin(value: float, int_bits: int, frac_bits: int) -> str:
    """Convert float to fixed-point binary string for VHDL."""
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    fixed_val = round(value * scale)
    
    # Handle negative values (two's complement)
    if fixed_val < 0:
        fixed_val = (1 << total_bits) + fixed_val
    
    # Clamp to valid range
    max_val = (1 << total_bits) - 1
    fixed_val = max(0, min(fixed_val, max_val))
    
    return f'"{fixed_val:0{total_bits}b}"'


def generate_vhdl_lut_package(
    sigmoid_config: YamlSigmoidConfig,
    int_bits: int,
    frac_bits: int,
    lut: list,
    output_file: str = "src/packages/sigmoid_lut_pkg.vhd"
):
    """Generate VHDL package file for sigmoid LUT."""
    high_bit = sigmoid_config.index_high
    low_bit = sigmoid_config.index_low
    
    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

use work.types.all;

package sigmoid_lut_pkg is
    -- LUT Configuration
    constant LUT_SIZE : integer := {sigmoid_config.lut_size};
    constant LUT_BITS : integer := {sigmoid_config.lut_bits};  -- k = log2(LUT_SIZE)
    
    -- Range Configuration: [-2^RANGE_BITS, 2^RANGE_BITS)
    constant RANGE_BITS : integer := {sigmoid_config.range_bits};  -- n, range is [-2^n, 2^n)
    
    -- Index extraction bounds (after MSB flip transformation)
    -- To get LUT index: resize with saturation, flip MSB, extract bits as unsigned
    constant INDEX_HIGH : integer := {high_bit};  -- = RANGE_BITS = n
    constant INDEX_LOW : integer := {low_bit};   -- = RANGE_BITS + 1 - LUT_BITS = n + 1 - k
    constant INDEX_WIDTH : integer := INDEX_HIGH - INDEX_LOW + 1;  -- = LUT_BITS = k

    -- LUT stores fixed-point values directly (sfixed format)
    -- Note: LUT[0] = 0.0 and LUT[LUT_SIZE-1] = 1.0 are forced for proper saturation
    type sigmoid_lut_type is array(0 to LUT_SIZE - 1) of sfixed_bus;

    constant SIGMOID_LUT : sigmoid_lut_type := (
"""
    for i, (_idx, _x, sig_val) in enumerate(lut):
        bin_val = float_to_fixed_bin(sig_val, int_bits, frac_bits)
        comma = "" if i == len(lut) - 1 else ","
        vhdl_code += f"        {i} => {bin_val}{comma}\n"

    vhdl_code += """    );

end package sigmoid_lut_pkg;
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print(f"Generated LUT package: {output_file}")
    return output_file


def update_makefile(makefile_path: str = "Makefile"):
    """Update Makefile to include sigmoid LUT package."""
    with open(makefile_path, 'r') as f:
        content = f.read()

    if 'sigmoid_lut_pkg.vhd' not in content:
        new_content = content.replace(
            'FILES =\tsrc/types.vhd',
            'FILES =\tsrc/types.vhd \t\t\\\n\t\tsrc/sigmoid_lut_pkg.vhd'
        )

        with open(makefile_path, 'w') as f:
            f.write(new_content)

        print(f"Updated {makefile_path} to include sigmoid_lut_pkg.vhd")
    else:
        print(f"{makefile_path} already includes sigmoid_lut_pkg.vhd")


def main():
    parser = argparse.ArgumentParser(
        description='Generate sigmoid LUT and package for VHDL'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (searches for *.nn_conf.yaml if not specified)')
    parser.add_argument('--root', type=str, default='.',
                        help='Project root directory (default: current directory)')

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    # Load configuration
    try:
        config = auto_load_config(root_dir, args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    print("=" * 60)
    print("Sigmoid Package Generation")
    print("=" * 60)
    print(f"LUT Size: {config.sigmoid.lut_size} entries (k = {config.sigmoid.lut_bits} bits)")
    print(f"Range: [-2^{config.sigmoid.range_bits}, 2^{config.sigmoid.range_bits}) = [{config.sigmoid.input_range[0]}, {config.sigmoid.input_range[1]})")
    print(f"Data Width: {config.data_width} bits ({config.int_bits} int, {config.frac_bits} frac)")
    print(f"Index extraction: bits ({config.sigmoid.index_high} downto {config.sigmoid.index_low})")
    print("=" * 60)

    # Generate LUT
    print("\n[1/2] Generating sigmoid LUT...")
    lut = generate_sigmoid_lut(config.sigmoid, config.frac_bits)

    # Generate VHDL LUT package
    print("\n[2/2] Generating VHDL LUT package...")
    output_file = root_dir / "src" / "packages" / "sigmoid_lut_pkg.vhd"
    generate_vhdl_lut_package(config.sigmoid, config.int_bits, config.frac_bits, lut, str(output_file))

    # Update Makefile
    print("\n[Optional] Updating Makefile...")
    update_makefile(str(root_dir / "Makefile"))

    print("\n" + "=" * 60)
    print("Package Generation Complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

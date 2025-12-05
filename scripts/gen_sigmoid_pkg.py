#!/usr/bin/env python3

import argparse
from pathlib import Path
from config import SigmoidConfig, sigmoid

def generate_sigmoid_lut(config : SigmoidConfig):
    lut = []
    x_min, x_max = config.input_range
    for i in range(config.lut_size):
        x = x_min + (x_max - x_min) * i / (config.lut_size - 1)
        sig_val = sigmoid(x)
        lut.append((i, x, sig_val))

    return lut

def generate_vhdl_lut_package(config : SigmoidConfig, lut, output_file="src/sigmoid_lut_pkg.vhd"):
    high_bit, low_bit = config.get_index_range()
    vhdl_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
    package sigmoid_lut_pkg is
        constant LUT_SIZE : integer := {config.lut_size};
        constant LUT_BITS : integer := {config.lut_bits};
        constant INPUT_WIDTH : integer := {config.data_width};
        constant OUTPUT_WIDTH : integer := {config.data_width};
        constant INDEX_HIGH : integer := {high_bit};
        constant INDEX_LOW : integer := {low_bit};
        constant INDEX_WIDTH : integer := INDEX_HIGH - INDEX_LOW + 1;
        constant INPUT_MAX : real := {config.input_range[1]};
        constant INPUT_MIN : real := {config.input_range[0]};
        constant SCALE_FACTOR : real := (real(LUT_SIZE) - 1.0) / (INPUT_MAX - INPUT_MIN);

        type sigmoid_lut_type is array(0 to LUT_SIZE - 1) of real;

        constant SIGMOID_LUT : sigmoid_lut_type := (
    """
    for i, (idx, x, sig_val) in enumerate(lut):
        if i == len(lut) - 1:
            vhdl_code += f"                {i} => {sig_val:.10f}\n"
        else:
            vhdl_code += f"        {i} => {sig_val:.10f},\n"

    vhdl_code += """        );

end package sigmoid_lut_pkg;
    """

    Path(output_file).parent.mkdir(parents = True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(vhdl_code)

    print("Generated LUT package file")
    return output_file




def update_makefile(makefile_path="Makefile"):
    """Update Makefile to include sigmoid LUT package"""

    with open(makefile_path, 'r') as f:
        content = f.read()

    # Check if sigmoid_lut_pkg is already in FILES
    if 'sigmoid_lut_pkg.vhd' not in content:
        # Add after types.vhd
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
    parser.add_argument('--lut-size', type=int, default=256,
                        help='Number of LUT entries (default: 256)')
    parser.add_argument('--data-width', type=int, default=32,
                        help='Data width in bits (default: 32)')
    parser.add_argument('--frac-bits', type=int, default=16,
                        help='Fractional bits (default: 16)')
    parser.add_argument('--input-min', type=float, default=-8.0,
                        help='Minimum input value (default: -8.0)')
    parser.add_argument('--input-max', type=float, default=8.0,
                        help='Maximum input value (default: 8.0)')

    args = parser.parse_args()

    # Create configuration
    config = SigmoidConfig(
        lut_size=args.lut_size,
        data_width=args.data_width,
        frac_bits=args.frac_bits,
        input_range=(args.input_min, args.input_max)
    )

    print("=" * 60)
    print("Sigmoid Package Generation")
    print("=" * 60)
    print(f"LUT Size: {config.lut_size} entries ({config.lut_bits} bits)")
    print(f"Input Width: {config.data_width} bits")
    print(f"Output Width: {config.data_width} bits")
    print("=" * 60)

    # Generate LUT
    print("\n[1/3] Generating sigmoid LUT...")
    lut = generate_sigmoid_lut(config)

    # Generate VHDL LUT package
    print("\n[2/3] Generating VHDL LUT package...")
    generate_vhdl_lut_package(config, lut)



    # Update Makefile
    print("\n[Optional] Updating Makefile...")
    update_makefile()

    print("\n" + "=" * 60)
    print("Package Generation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

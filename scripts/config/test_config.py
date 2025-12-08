#!/usr/bin/env python3
"""
Test script for YAML configuration loading and validation.
"""

import sys
from pathlib import Path

# Add scripts directory to path for proper import
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from config import load_config


def test_file(yaml_path: Path, should_pass: bool) -> bool:
    """Test a single YAML file. Returns True if test passed."""
    print(f"\nTesting: {yaml_path.name}")
    print("-" * 40)
    
    try:
        config = load_config(yaml_path)
        if should_pass:
            print(f"  [PASS] Loaded successfully")
            print(f"    Network: {config.network.name}")
            print(f"    Precision: {config.int_bits} int, {config.frac_bits} frac")
            print(f"    Sigmoid: {config.sigmoid.lut_size} LUT entries, range [-{2**config.sigmoid.range_bits}, {2**config.sigmoid.range_bits})")
            print(f"    Layers: {len(config.network.layers)}")
            return True
        else:
            print(f"  [FAIL] Expected validation error but loaded successfully")
            return False
    except (ValueError, FileNotFoundError) as e:
        if not should_pass:
            print(f"  [PASS] Correctly rejected with error:")
            # Print first few lines of error
            error_lines = str(e).split('\n')
            for line in error_lines[:5]:
                print(f"    {line}")
            if len(error_lines) > 5:
                print(f"    ... ({len(error_lines) - 5} more lines)")
            return True
        else:
            print(f"  [FAIL] Unexpected error: {e}")
            return False


def main():
    test_dir = Path(__file__).parent / "test_yaml"
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1
    
    print("=" * 50)
    print("Configuration Schema Tests")
    print("=" * 50)
    
    # Define test cases: (filename, should_pass)
    test_cases = [
        ("schema_1.yaml", True),   # Valid XOR network
        ("schema_2.yaml", True),   # Valid with total_bits
        ("false_schema.yaml", False),  # Invalid schema
    ]
    
    passed = 0
    failed = 0
    
    for filename, should_pass in test_cases:
        yaml_path = test_dir / filename
        if not yaml_path.exists():
            print(f"\nWarning: Test file not found: {filename}")
            continue
        
        if test_file(yaml_path, should_pass):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""
DEPRECATED: This module is deprecated. Use 'from config import ...' instead.

This file is kept for backward compatibility with gen_neural_network.py.
It will be removed once gen_neural_network.py is updated to use YAML config.
"""

import math
import warnings

warnings.warn(
    "scripts/config.py is deprecated. Use 'from config import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

def is_power_of_two(n):
    """Check if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0

class SigmoidConfig:
    """
    Configuration for sigmoid LUT generation with efficient hardware indexing.
    
    The indexing scheme works as follows:
    - Range is symmetric: [-2^n, 2^n) where range_exp = n
    - LUT size is 2^k where k = lut_bits
    - Total range span is 2^(n+1)
    - Step size is 2^(n+1) / 2^k = 2^(n+1-k)
    
    For efficient indexing (no arithmetic needed):
    1. Flip the MSB of the input to transform [-2^n, 2^n) -> [0, 2^(n+1))
    2. Extract bits (n downto n+1-k) as the unsigned index
    
    Constraint: k > n (lut_bits > range_exp) for fractional precision
    """
    def __init__(self, lut_size=256, data_width=32, frac_bits=16, num_test_inputs=16, range_exp=3):
        # Validate power-of-two constraints
        if not is_power_of_two(lut_size):
            raise ValueError(f"lut_size must be a power of two, got {lut_size}")
        if not is_power_of_two(data_width):
            raise ValueError(f"data_width must be a power of two, got {data_width}")
        
        self.lut_size = lut_size
        self.data_width = data_width
        self.frac_bits = frac_bits
        self.int_bits = data_width - frac_bits
        self.num_test_inputs = num_test_inputs
        
        # range_exp = n means range is [-2^n, 2^n)
        self.range_exp = range_exp
        self.input_range = (-2.0 ** range_exp, 2.0 ** range_exp)
        
        # k = log2(lut_size), number of bits needed to index the LUT
        self.lut_bits = lut_size.bit_length() - 1  # For power of two, this gives exact log2
        
        # Validate k > n constraint
        if self.lut_bits <= self.range_exp:
            raise ValueError(
                f"lut_bits ({self.lut_bits}) must be greater than range_exp ({self.range_exp}). "
                f"Increase lut_size or decrease range_exp."
            )
        
        # Calculate index extraction bounds
        # After MSB flip, we extract bits (n downto n+1-k) from the transformed value
        # n = range_exp, k = lut_bits
        # High bit: n (the bit just below the flipped sign bit position in transformed space)
        # Low bit: n + 1 - k (can be negative, meaning fractional bits)
        self.index_high = self.range_exp
        self.index_low = self.range_exp + 1 - self.lut_bits
        
        # Number of fractional bits needed for indexing
        # If index_low < 0, we need |index_low| fractional bits
        self.index_frac_bits = max(0, -self.index_low)
        
        # Number of upper bits to check for overflow detection
        # We compare input bits above range_exp with all-ones/all-zeros
        # For sfixed_bus, we check (INT_BITS-1 downto range_exp+1)
        self.overflow_check_bits = self.int_bits - 1 - self.range_exp

    def get_index_range(self):
        """Returns (high_bit, low_bit) for extracting LUT index from input."""
        return (self.index_high, self.index_low)
    
    def get_overflow_constants(self):
        """
        Returns info for overflow detection vectors.
        
        For positive overflow: upper bits (above range_exp) should all be 0 for valid positive numbers
        For negative overflow: upper bits (above range_exp) should all be 1 for valid negative numbers
        
        Returns dict with:
        - check_high: highest bit to check
        - check_low: lowest bit to check  
        - num_bits: number of bits in overflow check vector
        """
        return {
            'check_high': self.int_bits - 1,
            'check_low': self.range_exp + 1,
            'num_bits': self.overflow_check_bits
        }


def sigmoid(x):
    try:
        return 1.0/(1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

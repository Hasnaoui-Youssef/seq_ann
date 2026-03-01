"""Fixed-point conversion helpers for cocotb testbenches.

Converts between Python floats and the sfixed(INT_BITS-1 downto -FRAC_BITS) representation
used throughout the neural network accelerator.

Since GHDL VPI doesn't support indexing into sfixed_bus_array ports,
the cocotb_wrappers.vhd flattens array ports into wide std_logic_vectors.
These helpers handle the packing/unpacking.
"""

from pathlib import Path
import re


def _parse_types_vhd():
    """Parse DATA_WIDTH and FRAC_BITS from types.vhd."""
    types_path = Path(__file__).parent.parent / "src" / "packages" / "types.vhd"
    text = types_path.read_text()

    data_width = int(re.search(r"constant DATA_WIDTH\s*:\s*integer\s*:=\s*(\d+)", text).group(1))
    frac_bits = int(re.search(r"constant FRAC_BITS\s*:\s*integer\s*:=\s*(\d+)", text).group(1))
    return data_width, frac_bits


DATA_WIDTH, FRAC_BITS = _parse_types_vhd()
INT_BITS = DATA_WIDTH - FRAC_BITS


def real_to_sfixed(value: float) -> int:
    """Convert a real number to its sfixed bit representation (as a Python int).

    The result is a two's complement integer with DATA_WIDTH bits,
    representing sfixed(INT_BITS-1 downto -FRAC_BITS).
    """
    scaled = round(value * (1 << FRAC_BITS))
    mask = (1 << DATA_WIDTH) - 1
    return scaled & mask


def sfixed_to_real(bits: int) -> float:
    """Convert an sfixed bit pattern (DATA_WIDTH bits, two's complement) to a float."""
    bits = bits & ((1 << DATA_WIDTH) - 1)
    if bits >= (1 << (DATA_WIDTH - 1)):
        bits -= (1 << DATA_WIDTH)
    return bits / (1 << FRAC_BITS)


def set_sfixed(signal, value: float):
    """Set a cocotb signal to an sfixed value from a float."""
    signal.value = real_to_sfixed(value)


def get_sfixed(signal) -> float:
    """Read a cocotb signal as a float from its sfixed representation."""
    return sfixed_to_real(signal.value.to_unsigned())


def pack_sfixed_array(values: list[float]) -> int:
    """Pack a list of floats into a single wide bit vector.

    Element 0 occupies the MSB portion, element N-1 the LSB portion.
    This matches the cocotb_wrappers.vhd generate indexing.
    """
    result = 0
    for v in values:
        result = (result << DATA_WIDTH) | real_to_sfixed(v)
    return result


def unpack_sfixed_array(signal, count: int) -> list[float]:
    """Unpack a wide std_logic_vector signal into a list of floats.

    Element 0 is in the MSB portion, element N-1 in the LSB portion.
    """
    raw = signal.value.to_unsigned()
    result = []
    for i in range(count):
        shift = (count - 1 - i) * DATA_WIDTH
        mask = (1 << DATA_WIDTH) - 1
        element_bits = (raw >> shift) & mask
        result.append(sfixed_to_real(element_bits))
    return result

import math

class SigmoidConfig:
    def __init__(self, lut_size=256, data_width=32, frac_bits=16, num_test_inputs=16, input_range=(-8.0, 8.0)):
        self.lut_size = lut_size
        self.data_width = data_width
        self.frac_bits = frac_bits
        self.int_bits = data_width - frac_bits
        self.num_test_inputs = num_test_inputs
        self.input_range = input_range
        self.lut_bits = (lut_size - 1).bit_length()


    def get_index_range(self):
        if self.lut_bits <= self.int_bits:
            high_bit = (self.lut_bits // 2) - 1
            low_bit = - (self.lut_bits - (self.lut_bits // 2))
            return (high_bit, low_bit)

        high_bit = self.int_bits - 1
        low_bit = -self.frac_bits
        return (high_bit, low_bit)


def sigmoid(x):
    try:
        return 1.0/(1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

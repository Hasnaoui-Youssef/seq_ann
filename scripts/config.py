import math

class SigmoidConfig:
    def __init__(self, lut_size=256, input_width=32, input_frac_width=16, output_width=32, num_test_inputs=16, input_range=(-8.0, 8.0)):
        self.lut_size = lut_size
        self.input_width = input_width
        self.input_frac_width = input_frac_width
        self.output_width = output_width
        self.num_test_inputs = num_test_inputs
        self.input_range = input_range
        self.lut_bits = (lut_size - 1).bit_length()
        self.data_width = input_width // 2


    def get_index_range(self):
        if self.lut_bits <= self.data_width:
            high_bit = (self.lut_bits // 2) - 1
            low_bit = - (self.lut_bits - (self.lut_bits // 2))
            return (high_bit, low_bit)

        high_bit = (self.data_width // 2) - 1
        low_bit = - (self.data_width // 2)
        return (high_bit, low_bit)


def sigmoid(x):
    try:
        return 1.0/(1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

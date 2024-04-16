import torch
from torch import Tensor, nn
import optimized_bitlinear as obl
import torch.nn.functional as F


def weight_quant(w):
    """
    from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf,
    This is a little bit different from paper by adding '/ scale' in the end,
    which is super crucial for training (7.5 loss vs 2.5)
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinear(nn.Linear):
    """
    A modified version of bit linear, only apply bit quant to weight.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer, applying quantization to weights.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        w = self.weight
        w_quant = weight_quant(w)
        w_quant = w + (w_quant - w).detach()  # Apply quantization adjustments
        return F.linear(x, w_quant, self.bias)


class InferenceLinear(nn.Linear):
    """
    A modified version of bit linear, only apply bit quant to weight.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scale = None

    def quantize_weight(self):
        self.scale = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)
        quantized_weight = (self.weight * self.scale).round().clamp_(-1, 1).to(torch.int8)
        self.weight.requires_grad = False
        self.weight.data = quantized_weight

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer, applying quantization to weights.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        return obl.mat_mul(x / self.scale, self.weight, self.bias)

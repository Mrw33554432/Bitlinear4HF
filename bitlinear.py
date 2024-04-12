import torch.nn.functional as F
from torch import Tensor, nn


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
        w_quant = w + (weight_quant(w) - w).detach()  # Apply quantization adjustments
        return F.linear(x, w_quant, self.bias)

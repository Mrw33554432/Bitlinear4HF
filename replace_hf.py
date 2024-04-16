import gc

import torch
from torch import nn

from bitlinear import BitLinear, InferenceLinear


# Adapt from https://github.com/kyegomez/BitNet/blob/main/bitnet/replace_hf.py
def replace_linear_in_hf(model, keep_param: bool, custom_kernel=False):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear, except lm_head.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
        :param custom_kernel: default False, if true, it will use the custom kernel to replace the linear layer.
        :param model: The model to modify.
        :param keep_param: if ture, the model will keep param from the initial model.
        if false, the model will be using random init weight (For training)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if 'head' in name:
                continue
            # Create a new BitLinear layer with random parameters
            if not custom_kernel:
                bit_linear = BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                )
            else:
                bit_linear = InferenceLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                )

            if keep_param:
                # Transfer the weights and bias from the original nn.Linear to the new BitLinear
                data = module.weight.data
                bit_linear.weight.data.copy_(data)
                if module.bias is not None:
                    bit_linear.bias.data.copy_(module.bias.data)

            if custom_kernel:
                bit_linear.quantize_weight()

            del module

            # Replace the nn.Linear with the new BitLinear
            setattr(model, name, bit_linear)
        else:
            # Recursively apply to child modules
            replace_linear_in_hf(module, keep_param, custom_kernel=custom_kernel)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

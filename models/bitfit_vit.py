
import torch
import torch.nn as nn
from typing import List, Tuple

def apply_bitfit_to_vit(
    model: nn.Module,
    tune_layernorm_bias: bool = True,
    tune_mlp_bias: bool = True,
    tune_attn_bias: bool = True,
    tune_classifier: bool = True,
    verbose: bool = True,
) -> List[nn.Parameter]:
    """
    Enable BitFit (bias-only fine-tuning) for a ViT model from Hugging Face.

    This function:
      1) Freezes all parameters.
      2) Unfreezes only bias parameters (and optionally classifier head).
      3) Returns a list of trainable parameters for constructing your optimizer.

    Args:
        model: A Hugging Face ViT model (e.g., ViTForImageClassification).
        tune_layernorm_bias: Include LayerNorm.bias params.
        tune_mlp_bias: Include MLP Linear.bias params.
        tune_attn_bias: Include attention Linear.bias params.
        tune_classifier: Also train the classification head parameters.
        verbose: Print which parameters are made trainable.

    Returns:
        List of nn.Parameter objects to pass to the optimizer.
    """
    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    trainable: List[nn.Parameter] = []

    for full_name, param in model.named_parameters():
        if full_name.endswith(".bias"):
            param.requires_grad = True
            trainable.append(param)
            if verbose:
                print(f"[BitFit] Trainable bias: {full_name} ({param.numel()} params)")

    # 3) Optionally unfreeze classifier head
    if tune_classifier and hasattr(model, "classifier"):
        for n, p in model.classifier.named_parameters(recurse=True):
            p.requires_grad = True
            trainable.append(p)
            if verbose:
                print(f"[BitFit] Trainable (classifier): classifier.{n} shape={tuple(p.shape)}")
    else:
        # Some ViT variants use `model.vit` + `model.classifier` — already covered above
        pass

    if verbose:
        total = sum(p.numel() for p in trainable)
        print(f"[BitFit] Total trainable params: {total:,} across {len(trainable)} tensors")

    return trainable


def bitfit_param_groups(trainable_params: List[nn.Parameter]) -> Tuple[dict, dict]:
    """
    Typical practice is to set weight decay to 0 for biases. This helper returns
    two parameter groups: (no_decay for biases) and (decay for others).

    In classical BitFit, everything except biases is frozen, so the "decay" group
    will usually be empty unless you also train a classifier weight.

    Returns:
        (no_decay_group, decay_group) where each is a dict suitable for an optimizer.
    """
    bias_params = [p for p in trainable_params if p.ndim == 1]
    other_params = [p for p in trainable_params if p.ndim != 1]
    return (
        {"params": bias_params, "weight_decay": 0.0},
        {"params": other_params},  # use the optimizer's default weight_decay
    )

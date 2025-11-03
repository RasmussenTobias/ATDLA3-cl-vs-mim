import torch
import torch.nn as nn
from contextlib import contextmanager

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # base (frozen) weight/bias
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        # LoRA params
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x):
        # base path (frozen)
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        # lora path
        lora = self.dropout(x) @ self.A.t()    # [*, r]
        lora = lora @ self.B.t()               # [*, out_features]
        return out + self.scale * lora

def _replace_linear_with_lora(module: nn.Module, names, r=8, alpha=16, dropout=0.0):
    for name, child in module.named_children():
        if name in names and isinstance(child, nn.Linear):
            print(f"Applying LoRA to: {name}")
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            _replace_linear_with_lora(child, names, r=r, alpha=alpha, dropout=dropout)

def apply_lora_to_vit(model: nn.Module, r=8, alpha=16, dropout=0.0, target_modules=None):
    if target_modules is None:
        # Default target modules for Hugging Face ViT
        target_modules = ['query', 'key', 'value', 'dense']
    
    print("Model structure before LoRA:")
    print_model_structure(model)
    
    # swap target linears to LoRA
    _replace_linear_with_lora(model, target_modules, r=r, alpha=alpha, dropout=dropout)

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze LoRA params + classifier head
    trainable = []
    for n, p in model.named_parameters():
        # Look for LoRA parameters (A, B) and classifier parameters
        if any(k in n for k in ['A', 'B', 'classifier.weight', 'classifier.bias']):
            p.requires_grad = True
            trainable.append(p)
            print(f"Trainable parameter: {n}, shape: {p.shape}")
    
    print(f"Total trainable parameters: {len(trainable)}")
    return trainable

def print_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """Helper function to print model structure"""
    if current_depth >= max_depth:
        return
    
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            print(f"Linear layer: {full_name} -> in:{child.in_features}, out:{child.out_features}")
        elif hasattr(child, 'named_children') and len(list(child.named_children())) > 0:
            print(f"Module: {full_name}")
            print_model_structure(child, full_name, max_depth, current_depth + 1)

def count_lora_parameters(model):
    """Count LoRA and trainable parameters separately"""
    lora_params = 0
    classifier_params = 0
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if 'A' in name or 'B' in name:
                lora_params += param.numel()
                print(f"LoRA param: {name} -> {param.numel()} parameters")
            elif 'classifier' in name:
                classifier_params += param.numel()
                print(f"Classifier param: {name} -> {param.numel()} parameters")
            else:
                print(f"Other trainable param: {name} -> {param.numel()} parameters")
    
    print(f"\n📊 Parameter Summary:")
    print(f"   LoRA parameters: {lora_params:,}")
    print(f"   Classifier parameters: {classifier_params:,}")
    print(f"   Total trainable: {total_trainable:,}")
    
    return lora_params, classifier_params, total_trainable

@contextmanager
def only_lora_and_head(model):
    # ensure grads are only flowing to adapters + head
    for n, p in model.named_parameters():
        if p.requires_grad and not any(k in n for k in ['A', 'B', 'classifier.weight', 'classifier.bias']):
            p.requires_grad = False
    yield
import torch.nn as nn

class AdaptFormerAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, dim)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.up_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.act(self.down_proj(x)))

def apply_adaptformer_to_vit(model:nn.Module, bottleneck_dim=64):
    """
    Apply AdaptFormer to ViT
    Returns trainable parameters
    """
    def make_forward_with_adapter(blk):
        original_forward = blk.forward
        def forward(*args, **kwargs):
            x = original_forward(*args, **kwargs)
            if hasattr(blk, 'adapter') and blk.adapter is not None:
                x = x + blk.adapter(x)
            return x
        return forward

    for blk in model.vit.encoder.layer:
        if not hasattr(blk, 'adapter') or blk.adapter is None:
            hidden_dim = blk.output.dense.out_features
            blk.adapter = AdaptFormerAdapter(hidden_dim, bottleneck_dim)
        blk.forward = make_forward_with_adapter(blk)

    for p in model.parameters():
        p.requires_grad = False

    for blk in model.vit.encoder.layer:
        for p in blk.adapter.parameters():
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"AdaptFormer applied")
    return trainable_params


def count_adaptformer_parameters(model):
    adapter_params = 0
    classifier_params = 0
    total_trainable = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if 'adapter' in name:
                adapter_params += param.numel()
                print(f"Adapter param: {name} -> {param.numel()} parameters")
            elif 'classifier' in name or 'head' in name:
                classifier_params += param.numel()
                print(f"Classifier param: {name} -> {param.numel()} parameters")
            else:
                print(f"Other trainable param: {name} -> {param.numel()} parameters")

    print(f"Parameter Summary:")
    print(f"Adapter parameters: {adapter_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total trainable: {total_trainable:,}")

    return adapter_params, classifier_params, total_trainable
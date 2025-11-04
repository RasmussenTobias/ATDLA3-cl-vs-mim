import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
import os
import json
import pandas as pd  # Added for CSV export
from models.bitfit_vit import apply_bitfit_to_vit, bitfit_param_groups
from utils import get_dataset_config
from models.adapt_former import apply_adaptformer_to_vit, count_adaptformer_parameters
from models.lora_vit import apply_lora_to_vit, count_lora_parameters


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(dataset_name="flowers102", use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1, use_adaptformer=False, bottleneck_dim=64, 
          early_stopping=False, patience=10, use_bitfit=False):
    # with open("data/flowers102/cat_to_name.json", "r") as f:
    #     cat_to_name = json.load(f)
    dataset_config = get_dataset_config(dataset_name)
    data_dir = dataset_config["data_dir"]
    num_classes = dataset_config["num_classes"]

    mean = dataset_config["mean"]
    std = dataset_config["std"]
    
    batch_size = 4 if dataset_name == "flowers102" else 64
    num_epochs = 100
    lr = 5e-5
    weight_decay = 0.05
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    method = "lora" if use_lora else "adaptformer" if use_adaptformer else "bitfit" if use_bitfit else"full"

    method_dir = os.path.join(checkpoint_dir, dataset_name, method)
    os.makedirs(method_dir, exist_ok=True)
    
    print(f"Training on {dataset_name} dataset with {method} fine-tuning")
    print(f"Dataset: {num_classes} classes")
    if early_stopping:
        print(f"Early stopping enabled with patience: {patience}")

    training_history = []

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    elif dataset_name == "flowers102":
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ViTForImageClassification.from_pretrained(
        "facebook/dino-vits16",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    if use_lora:
        print(f"Using LoRA fine-tuning (r={lora_r}, alpha={lora_alpha})")
        trainable_params = apply_lora_to_vit(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        
        _, _, total_count = count_lora_parameters(model)
        
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        print(f"Training {len(trainable_params)} LoRA parameter tensors")
        print(f"Total trainable parameter count: {total_count:,}")    
    elif use_adaptformer:
        print("Using AdaptFormer fine-tuning")
        trainable_params = apply_adaptformer_to_vit(model, bottleneck_dim=bottleneck_dim)
        
        _, _, total_count = count_adaptformer_parameters(model)
        
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        print(f"Training {len(trainable_params)} AdaptFormer parameter tensors")
        print(f"Total trainable parameter count: {total_count:,}")  

    elif use_bitfit:
        print(f"Using BitFit fine-tuning")
        trainable_params = apply_bitfit_to_vit(model, tune_layernorm_bias=True, tune_mlp_bias=True, tune_attn_bias=True, tune_classifier=True, verbose=True)
        no_decay_groups, decay_group = bitfit_param_groups(trainable_params)
        
        param_groups = []
        if len(no_decay_groups["params"]) > 0:
            param_groups.append(no_decay_groups)
        if len(decay_group["params"]) > 0:
            decay_group.setdefault("weight_decay", weight_decay)
            param_groups.append(decay_group)

        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0)
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Training {total_params:,} parameters across {len(trainable_params)} tensors")

    else:
        print("Using full fine-tuning")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training {total_params:,} parameters")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        # Save every epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'dataset': dataset_name,
            'method': method,
            'is_best': val_acc > best_acc
        }
        training_history.append(epoch_data)
        
        history_file = os.path.join(method_dir, f"training_history_{dataset_name}_{method}.json")
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        df = pd.DataFrame(training_history)
        csv_file = os.path.join(method_dir, f"training_history_{dataset_name}_{method}.csv")
        df.to_csv(csv_file, index=False)
        
        # Best epoch and early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0  # Reset counter
            
            checkpoint_name = f"best_checkpoint_{dataset_name}_{method}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'dataset': dataset_name,
                'method': method,
                'lora_config': {'r': lora_r, 'alpha': lora_alpha, 'dropout': lora_dropout} if use_lora else None,
                'training_history': training_history
            }, os.path.join(method_dir, checkpoint_name))
            print(f"Checkpoint saved at epoch {epoch+1} with val_acc {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if early_stopping:
                print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
        
        # Early stopping check
        if early_stopping and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} consecutive epochs.")
            print(f"Best validation accuracy: {best_acc:.4f}")
            break

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")
    
    # Save final training summary
    summary = {
        'dataset': dataset_name,
        'method': method,
        'best_val_acc': best_acc,
        'total_epochs': len(training_history),
        'max_epochs': num_epochs,
        'early_stopping_used': early_stopping,
        'patience': patience if early_stopping else None,
        'stopped_early': len(training_history) < num_epochs if early_stopping else False,
        'final_train_acc': training_history[-1]['train_acc'],
        'final_val_acc': training_history[-1]['val_acc'],
        'config': {
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'lora_r': lora_r if use_lora else None,
            'lora_alpha': lora_alpha if use_lora else None,
            'lora_dropout': lora_dropout if use_lora else None
        }
    }
    
    summary_file = os.path.join(method_dir, f"training_summary_{dataset_name}_{method}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    
# To run main under different configurations: 
# python train.py
# python train.py --dataset cifar10 <- run full fine-tuning on cifar10
# python train.py --dataset flowers102 --use_lora  <- run LoRA fine-tuning on flowers102
# python train.py --dataset cifar10 --early_stopping --patience 7 <- with early stopping
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="flowers102", choices=["flowers102", "cifar10", "cifar100"])
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--use_adaptformer", action="store_true", help="Use AdaptFormer for fine-tuning")
    parser.add_argument("--bottleneck_dim", type=int, default=64, help="Bottleneck dimension for Adapt-former")
    parser.add_argument("--use_bitfit", action="store_true", help="Use BitFit fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    
    args = parser.parse_args()

    if (args.use_lora and args.use_bitfit) or (args.use_lora and  args.use_adaptformer) or (args.use_bitfit and args.use_adaptformer):
        raise SystemExit("Choose either --use_lora, --use_bitfit or --use_adaptformer (not multiple)")
    
    train(dataset_name=args.dataset,                                                    # Dataset
          use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,       # LoRA arguments
          use_adaptformer=args.use_adaptformer, bottleneck_dim=args.bottleneck_dim,     # AdaptFormer arguments
          use_bitfit=args.use_bitfit,                                                   # BitFit arguments
          early_stopping=args.early_stopping, patience=args.patience)                   # Early stopping arguments

import os
import requests
import torch
import numpy as np


def subsample(dataset, ratio, random=False):
    """
    Get indices of subsampled dataset with given ratio.
    """
    idxs = list(range(len(dataset)))
    idxs_sorted = {}
    for idx, target in zip(idxs, dataset.targets):
        if target in idxs_sorted:
            idxs_sorted[target].append(idx)
        else:
            idxs_sorted[target] = [idx]

    for idx in idxs_sorted:
        size = len(idxs_sorted[idx])
        lenghts = (int(size * ratio), size - int(size * ratio))
        if random:
            idxs_sorted[idx] = torch.utils.data.random_split(idxs_sorted[idx], lenghts)[0]
        else:
            idxs_sorted[idx] = idxs_sorted[idx][:lenghts[0]]

    idxs = [idx for idxs in idxs_sorted.values() for idx in idxs]
    return idxs




def download(url, path, force=False):
    from pathlib import Path
    from tqdm import tqdm

    # This snippet is based on https://stackoverflow.com/a/37573701
    
    if not force and os.path.exists(path):
        return
    
    # make dir
    root_path = "/".join(path.split("/")[:-1])
    if root_path != "":
        os.makedirs(root_path, exist_ok=True)
    
    # get url
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()



def restore(xs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean, std = np.array(mean), np.array(std)
    mean, std = mean.reshape([1, 3, 1, 1]), std.reshape([1, 3, 1, 1])
    return torch.clamp((xs * std) + mean, min=0.0, max=1.0)


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    configs = {
        "flowers102": {
            "data_dir": "./data/flowers102/dataset",
            "num_classes": 102,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "cifar10": {
            "data_dir": "./data/cifar10",
            "num_classes": 10,
            "mean": [0.485, 0.456, 0.406], 
            "std": [0.229, 0.224, 0.225]
        },
        "cifar100": {
            "data_dir": "./data/cifar-100-python",
            "num_classes": 100,
            "mean": [0.485, 0.456, 0.406], 
            "std": [0.229, 0.224, 0.225]
        },
        "imagenet": {
            "data_dir": "./data/imagenet",
            "num_classes": 1000,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(configs.keys())}")
    
    return configs[dataset_name]
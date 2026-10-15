import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union
import matplotlib.pyplot as plt
import pickle

import numpy as np
import torch
from path import Path
import os

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data"


def save_pkl(rounds, accuracy, loss, algo, dataset, model, cc):

    os.makedirs("results", exist_ok=True)

    out = {
        "rounds": rounds,
        "accuracy": accuracy,
        "loss": loss,
        "down_per_round": cc["down_per_round"],
        "up_per_round": cc["up_per_round"],
        "total_down": cc["total_down"],
        "total_up": cc["total_up"],

    }

    with open(f"results/{algo}_{dataset}_{model}.pkl", "wb") as f:
        pickle.dump(out, f)

def bytes_of(obj) -> int:
    if isinstance(obj, dict):
        tensors = obj.values()
    elif isinstance(obj, (list, tuple)):
        tensors = obj
    elif torch.is_tensor(obj):
        tensors = [obj]
    else:
        raise TypeError(f"Unsupported type for memory measure: {type(obj)}")

    return int(sum(t.numel() * t.element_size() for t in tensors))




def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )
    
def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=100)
    parser.add_argument("--local_steps", type=int, default=-1)
    parser.add_argument("--local_epochs", type=int, default=0)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--verbose_gap", type=int, default=5)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist"],
        default="mnist",
    )
    # parser.add_argument("--mime_beta", type=float, default=0.9)
    # parser.add_argument("--mime_full_grad_bs", type=int, default=1)  # 1=compute full local grad, 0=skip (approx/debug)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--server_lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.0)  # FedProx; 0 = FedAvg


    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target_acc", type=float, default=80.0)
    parser.add_argument("--client_num_per_round", type=int, default=2)
    parser.add_argument( "--model", type=str, choices=["lenet", "LG", "CNN", "VGG16", "RSN"], default="lenet")
    parser.add_argument("--save_period", type=int, default=20)
    return parser.parse_args()
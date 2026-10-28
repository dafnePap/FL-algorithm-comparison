import time
from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR)
sys.path.append(_CURRENT_DIR.parent)
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST

from constants import MEAN, STD
from partition import mixed_iid_sorted_partition
from utils.dataset import CIFARDataset, MNISTDataset

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "emnist": (EMNIST, MNISTDataset),
    "fmnist": (FashionMNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFARDataset),
}


def main(args):
    _DATASET_ROOT = (
        Path(args.root).abspath() / args.dataset
        if args.root is not None
        else _CURRENT_DIR.parent / args.dataset
    )
    _PICKLES_DIR = _CURRENT_DIR.parent / args.dataset / "pickles"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes_map = None

    target_transform = None

    if not os.path.isdir(_DATASET_ROOT):
        os.mkdir(_DATASET_ROOT)
    if os.path.isdir(_PICKLES_DIR):
        os.system(f"rm -rf {_PICKLES_DIR}")
    os.makedirs(_PICKLES_DIR, exist_ok=True)
    
    client_num_in_total = args.client_num_in_total
    client_num_in_total = args.client_num_in_total
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset( _DATASET_ROOT, train=True,download=True,split=args.emnist_split,transform=None)
        testset = ori_dataset( _DATASET_ROOT,train=False,split=args.emnist_split,transform=None)
    else:
        trainset = ori_dataset( _DATASET_ROOT, train=True, download=True)
        testset = ori_dataset(_DATASET_ROOT, train=False)
    concat_datasets = [trainset]
    # NOTE: s% iid + (1-s)% sorted-by-label
    # mixed IID
    all_datasets, stats = mixed_iid_sorted_partition(
        ori_dataset=concat_datasets,
        target_dataset=target_dataset,
        num_clients=client_num_in_total,
        dataset_name=args.dataset,
        s_iid=args.s_iid,
        transform=None,
        target_transform=None,
    )
       
    
    for subset_id, client_id in enumerate(
        range(0, len(all_datasets), args.client_num_in_each_pickles)
    ):
        subset = []
        for dataset in all_datasets[
            client_id : client_id + args.client_num_in_each_pickles
        ]:
            num_val_samples = int(len(dataset) * args.valset_ratio)
            num_test_samples = int(len(dataset) * args.test_ratio)
            num_train_samples = len(dataset) - num_val_samples - num_test_samples
            train, val, test = random_split(
                dataset, [num_train_samples, num_val_samples, num_test_samples]
            )
            subset.append({"train": train, "val": val, "test": test})
        with open(_PICKLES_DIR / str(subset_id) + ".pkl", "wb") as f:
            pickle.dump(subset, f)

    # save stats
    if args.type == "user":
        train_clients_num = int(client_num_in_total * args.fraction)
        clients_4_train = [i for i in range(train_clients_num)]
        clients_4_test = [i for i in range(train_clients_num, client_num_in_total)]

        with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
            pickle.dump(
                {
                    "train": clients_4_train,
                    "test": clients_4_test,
                    "total": client_num_in_total,
                },
                f,
            )

        train_clients_stats = dict(
            zip(clients_4_train, list(stats.values())[:train_clients_num])
        )
        test_clients_stats = dict(
            zip(
                clients_4_test,
                list(stats.values())[train_clients_num:],
            )
        )

        with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
            json.dump({"train": train_clients_stats, "test": test_clients_stats}, f)

    else:  # NOTE: "sample"  save stats
        client_id_indices = [i for i in range(client_num_in_total)]
        with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
            pickle.dump(
                {
                    "id": client_id_indices,
                    "total": client_num_in_total,
                },
                f,
            )
        with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
            json.dump(stats, f)

    args.root = (
        Path(args.root).abspath()
        if str(_DATASET_ROOT) != str(_CURRENT_DIR.parent / args.dataset)
        else None
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "emnist",
            "fmnist",
        ],
        default="mnist",
    )

    parser.add_argument("--client_num_in_total", type=int, default=10)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument("--valset_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--s_iid",
        type=float,
        default=0.0,
        help="Fraction of IID data in the mixed partition.",
    )
    parser.add_argument("--seed", type=int, default=int(time.time()))


    ################# For EMNIST only #####################
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    #######################################################
    parser.add_argument(
        "--type", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    parser.add_argument("--root", type=str, default="/root/repos/python/mine/datasets")
    args = parser.parse_args()
    main(args)
    args_dict = dict(args._get_kwargs())
    with open(_CURRENT_DIR.parent / "args.json", "w") as f:
        json.dump(args_dict, f)

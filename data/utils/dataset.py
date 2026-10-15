import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image

MEAN = {
    "mnist": 0.1307,
    "emnist": 0.1736,
    "fmnist": 0.2860,
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4865, 0.4409),
}
STD = {
    "mnist": 0.3015,
    "emnist": 0.3248,
    "fmnist": 0.3205,
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2009, 0.1984, 0.2023),
}
def _emnist_transpose(t):
    return t.transpose(1, 2)

def make_default_transform(dataset_name: str):
    ops = [transforms.ToTensor()]
    if dataset_name == "emnist":
        ops.append(transforms.Lambda(_emnist_transpose))  # named function, not lambda
    ops.append(transforms.Normalize(MEAN[dataset_name], STD[dataset_name]))
    return transforms.Compose(ops)
    
class MNISTDataset(Dataset):
    def __init__(self, data, targets, dataset_name, transform=None, target_transform=None, **kwargs):
        if dataset_name is None:
            raise ValueError("dataset_name is required")
        self.dataset_name = dataset_name
        self.images = data          # keep as raw numpy uint8
        self.labels = targets
        self.transform = transform if transform is not None else make_default_transform(dataset_name)
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)   # PIL path → ToTensor divides by 255 correctly
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.labels)

class CIFARDataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        dataset_name="cifar10",
        transform=None,
        target_transform=None,
    ):
        self.dataset_name = dataset_name
        self.images = np.asarray(data)
        self.labels = np.asarray(targets)

        self.transform = (
            transform
            if transform is not None
            else make_default_transform(dataset_name)
        )
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img = self.images[idx]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
import numpy as np
from path import Path
from rich.console import Console
from torch.utils.data import Subset, DataLoader

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_dataset


class ClientBase:
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_steps: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None
        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None
        self.model: torch.nn.Module = deepcopy(backbone).to(self.device)
        weight_decay = 5e-4 if dataset == "cifar10" else 0.0
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=local_lr, momentum=0.0 ,weight_decay=weight_decay,
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_steps = local_steps 
        self.local_lr = local_lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.untrainable_params: Dict[str, Dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def evaluate(self, use_valset=True):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = 0
        correct = 0
        dataloader = DataLoader(self.valset if use_valset else self.testset, 32)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)
            correct += (pred == y).int().sum()
        return loss.item(), correct.item()

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        evaluate=False,
        verbose=False,
        use_valset=True,
    ) -> Tuple[List[torch.Tensor], int]:
        self.client_id = client_id
        # Load the current global model and the selected client's local dataset
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        res, stats = self._log_while_training(evaluate, verbose, use_valset)()
        return res, stats



    def _train(self):
        self.model.train()
        n = len(self.trainset.dataset)

        # Decide mode:
        # - If local_epochs is set (>0), do epoch-based training using a DataLoader
        # - Otherwise, do step-based training 
        if self.local_epochs > 0:
            bs = self.batch_size if self.batch_size > 0 else 64  
            loader = DataLoader(self.trainset, batch_size=bs, shuffle=True, drop_last=False)
            #print(len(loader))
            steps_done = 0
            step_cap = None  

            for _ in range(self.local_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # Apply data augmentation only for CIFAR-10 training
                    if self.dataset == "cifar10":
                        x = self._augment(x)
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        else:
            # Step mode (exactly local_steps updates)
            local_steps = self.local_steps
            # batch size rule:
            for _ in range(local_steps):
                x, y = self.get_data_batch()  
                # Apply data augmentation only for CIFAR-10 training
                if self.dataset == "cifar10":
                    x = self._augment(x)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Return the locally updated model and the number of training samples
        return (
            [v.detach().clone() for v in self.model.state_dict().values()],
            len(self.trainset.dataset),
        )

    def test(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        loss, correct = self.evaluate()
        stats = {"loss": loss, "correct": correct, "size": len(self.testset)}
        return stats

    def get_client_local_dataset(self):
        datasets = get_dataset(
            self.dataset,
            self.client_id,
        )
        self.trainset = datasets["train"]
        self.valset = datasets["val"]
        self.testset = datasets["test"]

    def _log_while_training(self, evaluate=True, verbose=False, use_valset=True):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            num_samples = len(self.valset)
            if evaluate:
                loss_before, correct_before = self.evaluate(use_valset)

            res = self._train(*args, **kwargs)

            if evaluate:
                loss_after, correct_after = self.evaluate(use_valset)

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        loss_before / num_samples,
                        loss_after / num_samples,
                        correct_before / num_samples * 100.0,
                        correct_after / num_samples * 100.0,
                    )
                )

            stats = {
                "correct": correct_before,
                "size": num_samples,
            }
            return res, stats

        return _log_and_train

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def _augment(self, x):
        """RandomCrop(32, pad=4, reflect) + RandomHorizontalFlip on already-
        normalized CIFAR tensors. Only used during training."""
        B, C, H, W = x.shape
        pad = 4
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
        out = torch.empty_like(x)
        tops = torch.randint(0, 2 * pad + 1, (B,))
        lefts = torch.randint(0, 2 * pad + 1, (B,))
        flips = torch.rand(B) < 0.5
        for i in range(B):
            t, l = tops[i].item(), lefts[i].item()
            crop = x_padded[i, :, t:t + H, l:l + W]
            if flips[i]:
                crop = crop.flip(-1)
            out[i] = crop
        return out
    

    def get_data_batch(self):
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_steps)
        )
        
        # Randomly pick indices from the Subset length
        local_indices = np.random.choice(len(self.trainset), batch_size)
        #print(f"[Client {self.client_id}] Batch size used: {batch_size}")
        samples = []
        labels = []
        for idx in local_indices:
            x, y = self.trainset[idx] 
            samples.append(x)
            labels.append(y)
            
        return torch.stack(samples).to(self.device), torch.tensor(labels).to(self.device)
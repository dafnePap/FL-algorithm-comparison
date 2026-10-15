from collections import OrderedDict
from typing import OrderedDict, List

import torch
from rich.console import Console
from torch.utils.data import  DataLoader

from .base import ClientBase


class FedProxClient(ClientBase):
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
        mu: float = 0.0,
    ):
        super(FedProxClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_steps,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.mu = mu    
        print(f"Initialized FedProx client with mu={self.mu}")
   
    def _train(self):
        self.model.train()
        bs = self.batch_size if self.batch_size > 0 else 64  
        loader = DataLoader(self.trainset, batch_size=bs, shuffle=True, drop_last=False)
        #print(len(loader))
        steps_done = 0
        # Epoch mode
        if self.local_epochs > 0:
            bs = self.batch_size if self.batch_size > 0 else 64  
            loader = DataLoader(self.trainset, batch_size=bs, shuffle=True, drop_last=False)
            #print(len(loader))
            #print(f"Initialized FedProx client with mu={self.mu}")

            for _ in range(self.local_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if self.dataset == "cifar10":
                        x = self._augment(x)
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Add the FedProx proximal correction to the local gradient
                    for w, w0 in zip(self.model.parameters(), self.global_params):
                        w.grad.data += self.mu * (w0.data-w.data )
                    self.optimizer.step()
                    steps_done += 1

        else:
            # Step mode
            local_steps = self.local_steps
            # batch size rule:
            for _ in range(local_steps):
                x, y = self.get_data_batch()  
                if self.dataset == "cifar10":
                    x = self._augment(x)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                # Add the FedProx proximal correction to the local gradient
                for w, w0 in zip(self.model.parameters(), self.global_params):
                    w.grad.data += self.mu * (w0.data - w.data)
                self.optimizer.step()

        return (
            [v.detach().clone() for v in self.model.state_dict().values()],
            len(self.trainset.dataset),
        )

    def set_parameters(self, model_params: OrderedDict[str, torch.Tensor]):
        super().set_parameters(model_params)
        self.global_params = [p.detach().clone() for p in self.model.parameters()]
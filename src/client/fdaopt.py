# client/fdaopt.py
from collections import OrderedDict
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from .base import ClientBase


class FDAOptClient(ClientBase):
    """
    corresponds to fdaopt/training/fed_train.py 
    client does local SGD training
    """

    def train_client(self, client_id, model_params, epochs=0, steps=0, evaluate=False, verbose=False):
        self.client_id = client_id
        self.set_parameters(model_params) #Load client-specific params 
        self.model.train()
        self.get_client_local_dataset()
        # Perform one or more local SGD epochs or steps as requested by the server
        # Epoch mode
        if epochs > 0:
            bs = self.batch_size if self.batch_size > 0 else 64
            loader = DataLoader(self.trainset, batch_size=bs, shuffle=True, drop_last=False)

            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if self.dataset == "cifar10":
                        x = self._augment(x)
                    loss = self.criterion(self.model(x), y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # Step mode
        elif steps and steps > 0:  
            for _ in range(steps):
                x, y = self.get_data_batch()  
                if self.dataset == "cifar10":
                    x = self._augment(x)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        else:
            raise ValueError("train_work requires epochs>0 or steps>0")

        updated_list = [v.detach().clone() for v in self.model.state_dict().values()]
        weight = len(self.trainset)
        # Return the updated client model to the server

        return (updated_list, weight), {}

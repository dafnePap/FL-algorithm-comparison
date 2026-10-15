import torch
from rich.console import Console

from .base import ClientBase

# FedAvg uses the standard local SGD training defined in ClientBase
class FedAvgClient(ClientBase):
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
        super(FedAvgClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_steps,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )

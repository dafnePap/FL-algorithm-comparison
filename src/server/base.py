import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm
import time

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from config.models import LeNet5, LogisticRegression, CNN, VGG16, ResNet18
from config.util import (
    DATA_DIR,
    LOG_DIR,
    PROJECT_DIR,
    clone_parameters,
    fix_random_seed,
    save_pkl,
    bytes_of,
)

sys.path.append(PROJECT_DIR)
sys.path.append(DATA_DIR)
from client.base import ClientBase
from data.utils.util import get_client_id_indices
from data.utils.constants import MEAN, STD


class ServerBase:
    def __init__(self, args: Namespace, algo: str):
        self.algo = algo
        self.args = args
        self.log_name = "{}_{}_{}_{}.html".format( self.algo,self.args.dataset,self.args.global_epochs,self.args.local_epochs)
        self.device = torch.device(
            "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        )
        # Select model architecture
        if self.args.model == "LG":
            self.backbone = LogisticRegression
        elif self.args.model == "CNN":
            self.backbone = CNN
        elif self.args.model == "VGG16":
            self.backbone = VGG16
        elif self.args.model == "RSN":
            self.backbone = ResNet18
        else:
            self.backbone = LeNet5

        self.logger = Console(record=True,log_path=False,log_time=False)
        self.client_id_indices, self.client_num_in_total = get_client_id_indices(self.args.dataset )
        self.logger.log(f"The client num in total is {self.client_num_in_total}")
        self.logger.log(f"The client id indices is {self.client_id_indices}")
        self.results = { "accuracy": [],"loss": []}
        self.test_rounds = []
        self.comm = {"down_per_round": None,"up_per_round": None,"total_down": 0,"total_up": 0}
        fix_random_seed(self.args.seed) 

        self.server_model = self.backbone(self.args.dataset).to(self.device)
        self.global_params_dict = OrderedDict(self.server_model.state_dict(keep_vars=True))
        passed_epoch = 0
        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", self.server_model)
        self.global_test_loader = self.build_global_test_loader()
        self.trainer: ClientBase = None
       

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        self.start_time = time.perf_counter()
        self.time_to_target = None
        TARGET_ACC =  self.args.target_acc 

        progress_bar = (
            track(range(self.global_epochs),"[bold green]Training...",console=self.logger)
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training..."))
        globalm_bytes = bytes_of(self.global_params_dict)
        self.comm["down_per_round"] = globalm_bytes * self.args.client_num_per_round

        for E in progress_bar:
            self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)
            selected_clients = random.sample(self.client_id_indices, self.args.client_num_per_round) # Randomly select participating clients for the current  round
            res_cache = []
            print(selected_clients)
            # Send the current global model to each selected client and perform local training
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)

                res, stats = self.trainer.train(client_id=client_id, model_params=client_local_params,verbose=False)
                res_cache.append(res)
            
            if self.comm["up_per_round"] is None:
                updated_params_list = res_cache[0][0] 
                self.comm["up_per_round"] = bytes_of(updated_params_list) * self.args.client_num_per_round #res[0]-> updated_params_list

            # Keep a record of the communication cost of this round
            self.comm["total_down"] += self.comm["down_per_round"]
            self.comm["total_up"] += self.comm["up_per_round"]
            # Aggregate client models using a weighted average based on local dataset size
            self.aggregate(res_cache)
            # Evaluate the global model every two  rounds
            round_num = E + 1

            if round_num % 2 == 0:
                loss, acc = self.server_test()

                if acc >= TARGET_ACC and self.time_to_target is None:
                    self.time_to_target = time.perf_counter() - self.start_time
                    self.logger.log(
                        f"[TIME] Reached {TARGET_ACC}% accuracy in "
                        f"{self.time_to_target:.2f}s"
                    )

                    total_comm = self.comm["total_down"] + self.comm["total_up"]

                    self.logger.log(
                        f"[COMM@TARGET] down={self.comm['total_down']/1e6:.2f}MB | "
                        f"up={self.comm['total_up']/1e6:.2f}MB | "
                        f"total={total_comm/1e6:.2f}MB"
                    )

                self.results["loss"].append(loss)
                self.results["accuracy"].append(acc)
                self.test_rounds.append(round_num)

        save_pkl(self.test_rounds, self.results["accuracy"], self.results["loss"], self.algo,self.args.dataset, self.args.model, self.comm)
        self.logger.log( f"[COMM] Round {E} | "  f"Downlink: {self.comm['down_per_round'] / 1e6:.2f} MB | " f"Uplink: {self.comm['up_per_round'] / 1e6:.2f} MB | " f"Total: {(self.comm['total_down'] + self.comm['total_up']) / 1e6:.2f} MB" )

    @torch.no_grad()
    def aggregate(self, res_cache):
        updated_params_cache = list(zip(*res_cache))[0]
        weights_cache = list(zip(*res_cache))[1]
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        aggregated_params = []

        for params in zip(*updated_params_cache):
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )


    # Build a data test loader for evaluating the global model
    def build_global_test_loader(self):
        
        transform = transforms.Compose([
            transforms.ToTensor(), # Now it's a Tensor [1, 28, 28]
            # Swap dimension 1 (Height) and 2 (Width) only if it's EMNIST
            transforms.Lambda(lambda x: x.transpose(1, 2) if "emnist" in self.args.dataset else x),
            transforms.Normalize(MEAN[self.args.dataset], STD[self.args.dataset]),
        ])

        if self.args.dataset == "mnist":
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        elif self.args.dataset.startswith("emnist"):
            split = getattr(self.args, "emnist_split", "byclass")
            testset = datasets.EMNIST(root="./data", split=split, train=False, download=True, transform=transform)
        elif self.args.dataset == "cifar10":
            testset = datasets.CIFAR10(root="./data",train=False,download=True, transform=transform)
        
        return DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    @torch.no_grad()
    def server_test(self):
        self.logger.log("=" * 30, "SERVER GLOBAL TEST", "=" * 30, style="bold blue")

        model = self.backbone(self.args.dataset).to(self.device)
        model.load_state_dict(self.global_params_dict, strict=False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_sum, correct, n = 0.0, 0, 0

        x, y = next(iter(self.global_test_loader))
        self.logger.log(
            f"[SERVER TEST DEBUG] "
            f"min={x.min().item():.3f} "
            f"max={x.max().item():.3f} "
            f"mean={x.mean().item():.3f} "
            f"std={x.std().item():.3f}"
        )

        for x, y in self.global_test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            n += y.numel()

        avg_loss = loss_sum / n
        avg_acc = 100.0 * correct / n
        self.logger.log(f"[SERVER TEST] loss={avg_loss:.4f} acc={avg_acc:.2f}% (n={n})")
        return avg_loss, avg_acc


    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

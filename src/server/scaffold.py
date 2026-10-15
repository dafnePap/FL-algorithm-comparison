import pickle
import random

import torch
from rich.progress import track
from tqdm import tqdm
import time
from base import ServerBase
from client.scaffold import SCAFFOLDClient
from config.util import clone_parameters, get_args, save_pkl, bytes_of

# Initialize the Scaffold server and client trainer
class SCAFFOLDServer(ServerBase):
    def __init__(self):
        super(SCAFFOLDServer, self).__init__(get_args(), "SCAFFOLD")

        self.trainer = SCAFFOLDClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_steps=self.args.local_steps,  
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )
        # Initialize the global SCAFFOLD control variate to 0
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in self.backbone(self.args.dataset).parameters()
        ]
        self.global_lr = self.args.server_lr
        self.last_seen = {}


    def train(self):
        self.start_time = time.perf_counter()
        self.time_to_target = None
        global_bytes = bytes_of(self.global_params_dict)
        TARGET_ACC =  self.args.target_acc 
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        for E in progress_bar:
            self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)
            selected_clients = random.sample(self.client_id_indices, self.args.client_num_per_round)
            self.comm["down_per_round"] = (bytes_of(self.global_params_dict) +bytes_of(self.c_global)) * self.args.client_num_per_round
            res_cache = []
            gaps = []
            for cid in selected_clients:
                if cid in self.last_seen:
                    gaps.append(E - self.last_seen[cid])
                self.last_seen[cid] = E

            if gaps:
                print(f"[ROUND {E}] staleness: min={min(gaps)} max={max(gaps)}")
            else:
                print(f"[ROUND {E}] staleness: first-time clients only")

            for client_id in selected_clients:

                client_local_params = clone_parameters(self.global_params_dict)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    c_global=self.c_global,
                    verbose= False ,
                )
                res_cache.append(res)

                
            if self.comm["up_per_round"] is None:
                y_delta0, c_delta0 = res_cache[0]
                self.comm["up_per_round"] = (bytes_of(y_delta0) + bytes_of(c_delta0)) * self.args.client_num_per_round

            self.comm["total_down"] += self.comm["down_per_round"]
            self.comm["total_up"] += self.comm["up_per_round"]   
            # Aggregate client model updates
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


    def aggregate(self, res_cache):
        y_delta_cache = list(zip(*res_cache))[0]
        c_delta_cache = list(zip(*res_cache))[1]
        trainable_parameter = filter(
            lambda param: param.requires_grad, self.global_params_dict.values()
        )

        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.args.client_num_per_round
                for _ in range(self.args.client_num_per_round)
            ],
            device=self.device,
        )
        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.global_lr * x_del

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                self.args.client_num_per_round / len(self.client_id_indices)
            ) * c_del


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()

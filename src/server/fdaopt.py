# server/fdaopt.py
import random
from collections import OrderedDict
from typing import Dict, List
import time
import torch
from tqdm import tqdm
from rich.progress import track
from collections import deque
import pickle

from base import ServerBase
from client.fdaopt import FDAOptClient
from config.util import clone_parameters, get_args, save_pkl, bytes_of

from config.sketch import AmsSketch

@torch.no_grad()
def list_to_dict(template: OrderedDict, params_list: List[torch.Tensor]) -> OrderedDict:
    return OrderedDict(zip(template.keys(), params_list))

@torch.no_grad()
def copy_parameters_dict(src: OrderedDict, dst: OrderedDict):
    for k in dst.keys():
        dst[k].data.copy_(src[k].data)

@torch.no_grad()
def drift_list(round_start: OrderedDict, client_params: OrderedDict) -> List[torch.Tensor]:
    # Δ = w_client - w_start
    return [(client_params[k] - round_start[k]).detach() for k in round_start.keys()]

@torch.no_grad()
def vectorize(tensors: List[torch.Tensor], device) -> torch.Tensor:
    return torch.cat([t.to(device).view(-1) for t in tensors])

@torch.no_grad()
def compute_pseudo_gradients_from_drifts(client_drifts: Dict[int, List[torch.Tensor]], device) -> List[torch.Tensor]:
    # pseudo_gradients = - avg drift  (same as compute_pseudo_gradients) 
    cids = list(client_drifts.keys())
    avg = []
    for layer_tensors in zip(*[client_drifts[cid] for cid in cids]):
        stacked = torch.stack([t.to(device) for t in layer_tensors], dim=0)
        avg.append(torch.mean(stacked, dim=0))
    return [(-d) for d in avg]

@torch.no_grad()
def set_gradients(global_params_dict: OrderedDict, grads: List[torch.Tensor], device):
    for (k, p), g in zip(global_params_dict.items(), grads):
        p.grad = g.to(device)

@torch.no_grad()
def average_client_parameters(client_train_params_dict, device):
    num_clients = len(client_train_params_dict)
    average_params = None

    for train_params in client_train_params_dict.values():
        if average_params is None:
            average_params = [p.detach().to(device).clone() / num_clients for p in train_params]
        else:
            for p, avg_p in zip(train_params, average_params):
                avg_p.add_(p.detach().to(device) / num_clients)

    return average_params


@torch.no_grad()
def fda_sketch_estimation(client_drifts: Dict[int, List[torch.Tensor]], ams: AmsSketch, device) -> float:
    """
    Corresponds to original fda_sketch_estimation: compute avg drift vector, sketch it, return (1/(1+epsilon)) * M2(sketch)
    """
    avg_drift_list = average_client_parameters(client_drifts, device=device)
    avg_drift_vec = vectorize(avg_drift_list, device=device).float()

    sk = ams.sketch_for_vector(avg_drift_vec)
    epsilon = ams.epsilon
    est = (1.0 / (1.0 + epsilon)) * ams.estimate_euc_norm_squared(sk)
    return float(est)


@torch.no_grad()
def fda_variance_approx(client_drifts: Dict[int, List[torch.Tensor]], ams: AmsSketch, device) -> float:
    norm_sq_drifts = []
    for drifts in client_drifts.values():
        drift_vec = vectorize(drifts, device=device).float()
        norm_sq_drift = torch.dot(drift_vec, drift_vec)  # ||d_i||²
        norm_sq_drifts.append(norm_sq_drift)

    avg_norm_sq_drifts = sum(norm_sq_drifts) / len(norm_sq_drifts)# (1/K) Σ ||d_i||²

    norm_sq_avg_drift_approx = fda_sketch_estimation(client_drifts, ams, device=device) # ≈ || (1/K) Σ d_i ||² via sketch

    var_approx = avg_norm_sq_drifts - norm_sq_avg_drift_approx
    return float(var_approx.item() if hasattr(var_approx, "item") else var_approx)

@torch.no_grad()
def fda_variance_exact_terms(client_drifts, device):
    if not client_drifts: # Check if dictionary is empty
        return 0.0, 0.0, 0.0
    norm_sq = []
    for drifts in client_drifts.values():
        v = vectorize(drifts, device=device).double()
        norm_sq.append(torch.dot(v, v)) # ||d_i||²
    avg_norm_sq = sum(norm_sq) / len(norm_sq) # (1/K) Σ ||d_i||²

    avg_list = average_client_parameters(client_drifts, device=device)
    v_avg = vectorize(avg_list, device=device).double()
    norm_sq_avg = torch.dot(v_avg, v_avg)  # || (1/K) Σ d_i ||²

    var_exact = avg_norm_sq - norm_sq_avg
    return float(var_exact.item()), float(avg_norm_sq.item()), float(norm_sq_avg.item())



class FDAOptServer(ServerBase):
    def __init__(self):
        super().__init__(get_args(), "FDAOPT")

        self.trainer = FDAOptClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_steps=self.args.local_steps,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )

        self.theta = float("inf")        
        base_tau = self.args.local_epochs if (self.args.local_epochs>0) else self.args.local_steps
        self.max_units = base_tau * 3
        self.first_round = True
        self.theta_window = deque(maxlen=1)  # last 5 rounds
        self.theta_raw_history = []
        self.theta_smooth_history = []
        # server side model & optimizer 
        self.server_model = self.backbone(self.args.dataset).to(self.device)
        self.server_opt = torch.optim.Adam(self.server_model.parameters(),self.args.server_lr)

        # global weights snapshot 
        self.global_params_dict = OrderedDict(self.server_model.state_dict())

        # client cached params (like client_train_params_dict) 
        self.client_params_dict: Dict[int, OrderedDict] = {}
        self.ams = AmsSketch(depth=3, width=500)

    def _init_client_cache_for_round(self, sampled_clients, round_start):
        for cid in sampled_clients:
            if cid not in self.client_params_dict:# first time: create cache for client
                self.client_params_dict[cid] = clone_parameters(round_start)
            else: #if not first time: reset to  w_t (in-place)
                copy_parameters_dict(round_start, self.client_params_dict[cid])
    

    def train(self):
        self.start_time = time.perf_counter()
        self.time_to_target = None
        TARGET_ACC = self.args.target_acc 
        progress_bar = (
            track(range(self.global_epochs), "[bold green]Training.", console=self.logger)
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training.")
        )
        total_units = 0

        for r in progress_bar:
            sampled_clients = random.sample(self.client_id_indices, self.args.client_num_per_round)
            round_start = clone_parameters(self.global_params_dict)     # 1) save w_t at start of round 
            self._init_client_cache_for_round(sampled_clients, round_start)   # 2) reset each sampled client's cached params to round_start
            print("sampled_clients:", sampled_clients[:10], "unique:", len(set(sampled_clients)), "len:", len(sampled_clients))

            var_approx = 0.0
            s_t = 0
            client_drifts: Dict[int, List[torch.Tensor]] = {}
            self.logger.log(f"------var_approx: {var_approx},  self.theta: {self.theta}, round={r}, max epoch={ self.max_units})")
            self.comm["total_down"] += bytes_of(round_start) *len(sampled_clients)
            while var_approx <= self.theta and s_t < self.max_units:
                s_t += 1
                total_units +=1
                # train each sampled client for ONE more local epoch
                for cid in sampled_clients:
                    if self.args.local_epochs>0:
                        # one local epoch
                        res, _stats = self.trainer.train_client( client_id=cid,model_params=self.client_params_dict[cid], epochs=1, steps=0, evaluate=False, verbose=False)
                    else:
                        # one local step (one minibatch update)
                        res, _stats = self.trainer.train_client( client_id=cid,model_params=self.client_params_dict[cid],epochs=0, steps=1, evaluate=False,verbose=False)
                    updated_list, _weight = res
                    self.client_params_dict[cid] = list_to_dict(self.global_params_dict, updated_list)
                
                self.comm["total_up"] += self.ams.sketch_nbytes() * len(sampled_clients)
                # compute drifts Δ_k = w_k - w_t
                client_drifts = {cid: drift_list(round_start, self.client_params_dict[cid]) for cid in sampled_clients}

                #Calculate the current variance approximation 
                var_approx = fda_variance_approx(client_drifts, self.ams, device=self.device)

                # if self.first_round:
                #     self.first_round = False
                #     break
        
            self.comm["total_up"] += bytes_of(round_start) * len(sampled_clients)
            if s_t >= self.max_units:
                self.logger.log(f"[FDA] Round {r}: stop=MAX_EPOCHS ({s_t}/{self.max_units})")
            else:
                self.logger.log(f"[FDA] Round {r}: stop=THETA (var={var_approx:.2e}, theta={self.theta:.2e})")

            # server update via pseudo-gradients
            self.server_model.load_state_dict(round_start, strict=False) # load w_t to server model
            pseudo_grads = compute_pseudo_gradients_from_drifts(client_drifts, device=self.device)# calculate pseudo-gradients
            self.server_opt.zero_grad() 
            for p, g in zip(self.server_model.parameters(), pseudo_grads):
                p.grad = g.to(self.device)   # set pseudo-gradients
            self.server_opt.step()
            self.global_params_dict = OrderedDict(self.server_model.state_dict()) # update w_{t+1} from server model

            var_exact, _, _ = fda_variance_exact_terms(client_drifts, device=self.device)
           
            self.logger.log(f"[DBG] end round: var_approx={var_approx:.2e} var_exact={var_exact:.2e} theta={self.theta:.2e} s_t={s_t}")
            self.logger.log(f"Steps s_t={s_t} happenned")

            theta_raw = ((self.max_units / 2.0) / max(1, s_t)) * float(var_exact)# theta update

            self.theta_window.append(theta_raw)
            self.theta = sum(self.theta_window) / len(self.theta_window)

            self.theta_raw_history.append(theta_raw)
            self.theta_smooth_history.append(self.theta)
            # Evaluate the global model every two  rounds
            round_num = r + 1

            if round_num % 2 == 0:
                loss, acc = self.server_test()

                if acc >= TARGET_ACC and self.time_to_target is None:
                    self.time_to_target = time.perf_counter() - self.start_time
                    self.logger.log(
                        f"[TIME] Reached {TARGET_ACC}% accuracy in "
                        f"{self.time_to_target:.2f}s (units {total_units})"
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
                
        theta_file = f"results/theta_{self.algo}_{self.args.dataset}_{self.args.model}.pkl"
        with open(theta_file, "wb") as f:
            pickle.dump({
                "theta_raw": self.theta_raw_history,
                "theta_smooth": self.theta_smooth_history
            }, f)

        save_pkl(self.test_rounds, self.results["accuracy"], self.results["loss"], self.algo,self.args.dataset, self.args.model, self.comm)
        self.logger.log(f"Total num of total_units{total_units} ")


if __name__ == "__main__":
    server = FDAOptServer()
    server.run()

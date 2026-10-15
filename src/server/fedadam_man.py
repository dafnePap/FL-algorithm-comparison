from collections import OrderedDict
import torch
import time

from base import ServerBase
from client.fedavg import FedAvgClient
from config.util import get_args


class FedAdamManualServer(ServerBase):
    def __init__(self):
        super().__init__(get_args(), "FedAdam")

        self.trainer = FedAvgClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_steps=self.args.local_steps,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )

        # Paper/Flower-style hyperparams
        self.eta =  self.args.eta       # Flower uses eta /server lr ~ 1e-1 
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.tau = self.args.tau

        # Round counter for bias correction
        self.t = 0
        self.m_t = None
        self.v_t = None


    @torch.no_grad()
    def aggregate(self, res_cache):
        print("hhh")

        # server_round count for bias correction
        self.server_model.load_state_dict(self.global_params_dict, strict=False)

        self.t += 1       
        updated_params_cache = list(zip(*res_cache))[0]
        weights_cache = list(zip(*res_cache))[1]
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        # FedAvg aggregate (full state_dict order) 
        w_avg_list = []
        for layer_params in zip(*updated_params_cache):
            w_avg_list.append(torch.sum(weights * torch.stack(layer_params, dim=-1), dim=-1))

        keys = list(self.global_params_dict.keys())
        w_avg_state = OrderedDict(zip(keys, w_avg_list))
        
        param_names = [name for name, _ in self.server_model.named_parameters()]
        current_weights = [self.global_params_dict[name].detach() for name in param_names]
        fedavg_weights = [w_avg_state[name].detach() for name in param_names]
       
        # Adam delta: delta = fedavg - current  (same as Flower)
        delta_t = [x - y for x, y in zip(fedavg_weights, current_weights)]
       
        # m_t init/update
        if self.m_t is None:
            self.m_t = [torch.zeros_like(x) for x in delta_t]
        self.m_t = [self.beta1 * m + (1 - self.beta1) * d for m, d in zip(self.m_t, delta_t)]

        # v_t init/update
        if self.v_t is None:
            self.v_t = [torch.zeros_like(x) for x in delta_t]
        self.v_t = [self.beta2 * v + (1 - self.beta2) * (d * d) for v, d in zip(self.v_t, delta_t)]

        # bias-corrected lr: identical idea to Flower's version
        eta_norm = (
            self.eta * (1.0 - self.beta2 ** self.t) ** 0.5 / (1.0 - self.beta1 ** self.t)
        )

        new_weights = [
            w + eta_norm * m / (torch.sqrt(v) + self.tau)
            for w, m, v in zip(current_weights, self.m_t, self.v_t)
        ]
        
        # Write back updated params into the global state_dict ----
        for name, w_new in zip(param_names, new_weights):
            self.global_params_dict[name] = w_new

    @torch.no_grad()
    def aggregate_pytorch(self, res_cache):
        # Sync model to current global snapshot
        self.server_model.load_state_dict(self.global_params_dict, strict=False)

        self.t += 1
        updated_params_cache = list(zip(*res_cache))[0]
        weights_cache = list(zip(*res_cache))[1]
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        # FedAvg aggregate in state_dict order
        w_avg_list = []
        for layer_params in zip(*updated_params_cache):
            w_avg_list.append(torch.sum(weights * torch.stack(layer_params, dim=-1), dim=-1))

        keys = list(self.global_params_dict.keys())
        w_avg_state = OrderedDict(zip(keys, w_avg_list))

        # Parameters only (match torch.optim.Adam)
        param_names = [name for name, _ in self.server_model.named_parameters()]
        current_weights = [self.global_params_dict[name].detach() for name in param_names]
        fedavg_weights = [w_avg_state[name].detach() for name in param_names]
        grad_t = [w - w_f for w, w_f in zip(fedavg_weights, current_weights)]

        # Update moments 
        if self.m_t is None:
            self.m_t = [torch.zeros_like(g) for g in grad_t]
        if self.v_t is None:
            self.v_t = [torch.zeros_like(g) for g in grad_t]

        self.m_t = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m_t, grad_t)]
        self.v_t = [self.beta2 * v + (1 - self.beta2) * (g * g) for v, g in zip(self.v_t, grad_t)]

        # Bias-corrected moments (PyTorch Adam)
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m_t]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v_t]
        eta_norm = (
            self.eta * (1.0 - self.beta2 ** self.t) ** 0.5/ (1.0 - self.beta1 ** self.t)
        )

        # PyTorch-style update
        new_weights = [
            w + self.eta * mh / (torch.sqrt(vh) + self.tau)
            for w, mh, vh in zip(current_weights, m_hat, v_hat)
        ]

        # Write back updated params
        for name, w_new in zip(param_names, new_weights):
            self.global_params_dict[name] = w_new

        
if __name__ == "__main__":
    server = FedAdamManualServer()
    server.run()

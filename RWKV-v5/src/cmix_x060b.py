import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *

class RWKV_CMix_x060b(JITModClass):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        GATE_EXP_DIM = max(64, args.n_embd // 16)
        self.receptance = DDLorExp(in_dim=args.n_embd, hidden_dim=GATE_EXP_DIM, out_dim=args.n_embd)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @JITModMethod
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return self.receptance(xr) * kv

class DDLorExp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.w1 = nn.Parameter(torch.empty(in_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.w2 = nn.Parameter(torch.zeros(hidden_dim, out_dim))
        self.gmult = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glora = (torch.tanh(x @ self.w1) @ self.w2).exp()
        gate = self.gmult.to(glora) * glora
        return gate

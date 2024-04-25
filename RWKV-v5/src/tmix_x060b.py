import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *

class RWKV_Tmix_x060b(JITModClass):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.layer_id = layer_id
        self.dim_ffn = args.n_embd * 2
        self.dim_k = args.n_embd
        self.dim_v = args.n_embd

        self.k_head_size = args.head_size_a
        self.v_head_size = int(args.head_size_a * self.dim_v / self.dim_k)
        self.n_head = args.dim_att // self.k_head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for k,v,r
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.time_v_bonus = nn.Parameter(torch.full([self.dim_v], 2.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.dim_k, bias=False) # DK params
        self.key = nn.Linear(args.n_embd, self.dim_k, bias=False) # DK params
        self.v_ffn_gate = nn.Linear(args.n_embd, self.dim_v + self.dim_ffn + (self.dim_v + self.dim_ffn), bias=False) # 2D(V+F) params
        self.output = nn.Linear(self.dim_v + self.dim_ffn, args.n_embd, bias=False) # D(V+F) params
        self.ln_x = nn.LayerNorm(args.dim_att)

    @JITModMethod
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 3, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(3, B, T, -1)
        mk, mv, mr = xxx.unbind(dim=0)

        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)

        r = self.receptance(xr)
        k = (-self.key(xk).exp()).exp()
        v, ffn, g = self.v_ffn_gate(xv).split([self.dim_v, self.dim_ffn, self.dim_v+self.dim_ffn], dim=-1)
        v = v.contiguous()
        w = 1.0 - k
        u = torch.zeros(self.n_head, self.k_head_size, device=r.device, dtype=r.dtype)

        # FIXME - GQA

        # FIXME - support different rk, v sizing
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u)

        # v bonus
        x = x + r @ k.mT @ (self.time_v_bonus * v)

        x = self.ln_x(x)
        x = torch.cat([x, ffn], dim=-1)
        x = x * F.silu(g)
        x = self.output(x)
        return x

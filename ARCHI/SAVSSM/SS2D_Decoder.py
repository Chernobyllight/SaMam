import torch
import math
import torch.nn as nn
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from ARCHI.SAVSSM.common.SAIN import SRAdaIN
from ARCHI.SAVSSM.common.SConv import SConv
from ARCHI.archi_utils import get_permute_order


class SS2D(nn.Module):
    def __init__(
            self,
            d_model=64,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            representation_dim=64,
            mamba_from_trion=1,
            bias=False,
            device=None,
            dtype=None,
            zero_init=0
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if mamba_from_trion:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        else:
            from ARCHI.selective_scan_torch import selective_scan_fn

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.representation_dim = representation_dim

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        # self.conv2d = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     padding=(d_conv - 1) // 2,
        #     **factory_kwargs,
        # )
        self.SConv = SConv(in_channels=self.d_inner,
                           out_channels=self.d_inner,
                           kernel_size=3,
                           groups=self.d_inner,
                           representation_dim=representation_dim)

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        # self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        # self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.A_logs_generate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.representation_dim,
                      self.d_inner * 4 * self.d_state,
                      kernel_size=1)
        )
        self.Ds_generate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.representation_dim,
                      self.d_inner * 4,
                      kernel_size=1)
        )
        # self.A_logs_generate = nn.Linear(self.representation_dim, self.d_inner * 4 * self.d_state, bias=False)
        # self.Ds_generate = nn.Linear(self.representation_dim, self.d_inner * 4, bias=False)

        self.selective_scan = selective_scan_fn

        # self.out_norm = nn.LayerNorm(self.d_inner)
        self.SAIN = SRAdaIN(in_channels=self.d_inner, representation_dim=representation_dim, zero_init=zero_init)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    def forward_core_S7(self, x: torch.Tensor, representation):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x = x.view(B, C, -1).contiguous()
        (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4) = get_permute_order(H, W)
        # # print(scan_order)
        xs = torch.stack([x[:, :, o1], x[:, :, o2], x[:, :, o3], x[:, :, o4]], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        # Ds = self.Ds.float().view(-1)
        Ds = self.Ds_generate(representation).view(-1)
        # As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        As = -torch.exp(self.A_logs_generate(representation).float()).view(-1, self.d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y1 = out_y[:, 0, :, o1_inverse]
        y2 = out_y[:, 1, :, o2_inverse]
        y3 = out_y[:, 2, :, o3_inverse]
        y4 = out_y[:, 3, :, o4_inverse]

        return y1, y2, y3, y4

    def forward(self, x: torch.Tensor, representation):
        B, H, W, C = x.shape
        # xz = self.in_proj(x)
        # x, z = xz.chunk(2, dim=-1)
        x = self.in_proj(x)
        # B, H, W, C = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()  # B,H,W,C->B,C,H,W
        x = self.act(self.SConv(x, representation))

        y1, y2, y3, y4 = self.forward_core_S7(x, representation)

        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # y -> B,C,L
        y = y.view(B, -1, H, W).contiguous()
        y = self.SAIN(y, representation)  # y -> B,C,H,W

        y = y.permute(0, 2, 3, 1).contiguous()  # y: B,C,H,W -> B,H,W,C
        # y = y * F.silu(z)
        out = self.out_proj(y)  # out: B,H,W,C
        return out

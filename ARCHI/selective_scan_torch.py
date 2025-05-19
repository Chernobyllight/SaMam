import torch

def selective_scan_fn(
        u: torch.Tensor, # (B, K * C, L)
        delta: torch.Tensor, # (B, K * C, L)
        A: torch.Tensor, # (K * C, N)
        B: torch.Tensor, # (B, K, N, L)
        C: torch.Tensor, # (B, K, N, L)
        D: torch.Tensor = None, # (K * C)
        delta_bias: torch.Tensor = None, # (K * C)
        delta_softplus=True,
        oflex=True,
        *args,
        **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()

    # from einops import rearrange, repeat
    # B2 = repeat(B, "B G N L -> B (G H) N L", H=A.shape[0] // B.shape[1])
    # print(A.shape[0])
    # deltaB_u2 = torch.einsum('bdl,bdnl,bdl->bdln', delta, B2, u)

    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    # print(deltaB_u2 == deltaB_u)
    # exit(123)

    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)

    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)

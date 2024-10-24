import torch
from typing import Optional


def naive_recurrent_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    u_2d: bool = False
):
    torch_dtype = q.dtype if q.dtype in [torch.float16, torch.float32, torch.float64] else torch.float32
    orig_dtype = q.dtype
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, u = (x.to(dtype=torch_dtype) for x in (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state.to(dtype=torch_dtype)

    w = w.exp()

    if u_2d:
        u_expand = u[None, ..., None]
    else:
        u_expand = u[..., None]

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i] * scale
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u_expand * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i

    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


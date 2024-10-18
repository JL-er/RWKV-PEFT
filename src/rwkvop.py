import os
import torch
from einops import rearrange
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6


RUN_WKV6_GENERAL = None
RUN_WKV5_GENERAL = None
########################################################################################################
# CUDA Kernel
########################################################################################################
scale_factor = 1.0 if os.environ["RWKV_FLOAT_MODE"] != "fp16" else -1.0


def RUN_CUDA_RWKV6_STATE_FLA(B, T, C, H, r, k, v, w, u, s=None):
    r = rearrange(r, 'b l (h d) -> b h l d', h=H)
    k = rearrange(k, 'b l (h d) -> b h l d', h=H)
    v = rearrange(v, 'b l (h d) -> b h l d', h=H)
    w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h=H)
    o, state = chunk_rwkv6(r, k, v, w, u=u, scale=scale_factor, initial_state=s, output_final_state=True)
    x = rearrange(o, 'b h l d -> b l (h d)')
    return x, state


if os.environ["WKV"] == 'fla':
    if 'x060' in os.environ["RWKV_MY_TESTING"]:
        RUN_WKV6_GENERAL = RUN_CUDA_RWKV6_STATE_FLA
    else:
        # 'fla only supports x060'
        raise NotImplementedError('fla only supports x060')


else:
    from torch.utils.cpp_extension import load

    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

    if 'x060' in os.environ["RWKV_MY_TESTING"]:
        if (os.environ["RWKV_TRAIN_TYPE"] == 'infctx') or (os.environ["RWKV_TRAIN_TYPE"] == 'states'):
            if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                wkv6state_cuda = load(name="wkv6infctx", sources=["cuda/wkv6infctx_op.cpp", f"cuda/wkv6infctx_cuda.cu"],
                                      verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            else:
                wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                                      verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])

            class WKV_6STATE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                    with torch.no_grad():
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        assert s.dtype == torch.bfloat16
                        assert HEAD_SIZE == C // H
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        assert s.is_contiguous()
                        ctx.save_for_backward(r, k, v, w, u, s)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, w, u, s = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gs = torch.empty((B, H, C // H, C // H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                        gu = torch.sum(gu, 0).view(H, C // H)
                        gs = torch.sum(gs, 0).view(H, C // H, C // H)
                        return (None, None, None, None, gr, gk, gv, gw, gu, gs)

            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                x = WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
                return x, s
            RUN_WKV6_GENERAL = RUN_CUDA_RWKV6_STATE
        else:
            wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                             verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])

            class WKV_6_NO_STATE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, H, r, k, v, w, u):
                    with torch.no_grad():
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        assert HEAD_SIZE == C // H
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        ew = (-torch.exp(w.float())).contiguous()
                        ctx.save_for_backward(r, k, v, ew, u)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, ew, u = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                        gu = torch.sum(gu, 0).view(H, C // H)
                        return (None, None, None, None, gr, gk, gv, gw, gu)

            def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u, s=None):
                return WKV_6_NO_STATE.apply(B, T, C, H, r, k, v, w, u), None

            RUN_WKV6_GENERAL = RUN_CUDA_RWKV6
    else:
        wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                         verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])

        class WKV_5(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ew = (-torch.exp(w.float())).contiguous()
                    eew = (torch.exp(ew)).contiguous()
                    ctx.save_for_backward(r, k, v, eew, ew, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, eew, ew, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
                    wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                    gw = torch.sum(gw, 0).view(H, C // H)
                    gu = torch.sum(gu, 0).view(H, C // H)
                    return (None, None, None, None, gr, gk, gv, gw, gu)

        def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
            return WKV_5.apply(B, T, C, H, r, k, v, w, u)

        RUN_WKV5_GENERAL = RUN_CUDA_RWKV5

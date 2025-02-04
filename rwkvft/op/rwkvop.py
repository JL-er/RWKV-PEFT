
from einops import rearrange
import os, math, gc, importlib
import torch
########################################################################################################
# CUDA Kernel
########################################################################################################
def RUN_CUDA_RWKV7g():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_RWKV7_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV6():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV6_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV5():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


if os.environ["WKV"] == 'fla':
    if 'x070' in os.environ["RWKV_MY_TESTING"]:
        from fla.ops.rwkv7 import chunk_rwkv7
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx' or os.environ["RWKV_TRAIN_TYPE"] == 'states':
            def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                w=-torch.exp(w)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
                return o, state
        else:
            def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64): #compatible with cuda implement
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                w=-torch.exp(w)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False, head_first=False)
                return o
    from fla.ops.rwkv6 import chunk_rwkv6
    if 'x060' in os.environ["RWKV_MY_TESTING"]:
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x, state
        elif os.environ["RWKV_TRAIN_TYPE"] == 'states':
            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                s = s.transpose(1, 2).expand(B,*s.shape)
                o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=False)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x
        else:
            def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=None, output_final_state=False)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x

else:
    from torch.utils.cpp_extension import load

    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    if 'x070' in os.environ["RWKV_MY_TESTING"]:
        CHUNK_LEN = 16

        flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
        load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

        class WindBackstepping(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w,q,k,v,z,b):
                B,T,H,C = w.shape 
                assert T%CHUNK_LEN == 0
                assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
                assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
                y = torch.empty_like(v)
                s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
                ctx.save_for_backward(w,q,k,v,z,b,s,sa)
                return y
            @staticmethod
            def backward(ctx, dy):
                assert all(i.dtype==torch.bfloat16 for i in [dy])
                assert all(i.is_contiguous() for i in [dy])
                w,q,k,v,z,b,s,sa = ctx.saved_tensors
                dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
                torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
                return dw,dq,dk,dv,dz,db

        def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
            B,T,HC = q.shape
            q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
            return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

    elif 'x060' in os.environ["RWKV_MY_TESTING"]:
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            wkv6state_cuda = load(name="wkv6infctx", sources=["cuda/wkv6infctx_op.cpp", f"cuda/wkv6infctx_cuda.cu"],
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
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
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
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        gs = torch.sum(gs, 0).view(H, C//H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu, gs)

            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                x = WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
                return x, s
        elif os.environ["RWKV_TRAIN_TYPE"] == 'states':
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
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
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
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        gs = torch.sum(gs, 0).view(H, C//H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu, gs)

            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)

        else:
            wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
                
            class WKV_6(torch.autograd.Function):
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
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
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
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu)

            def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
                return WKV_6.apply(B, T, C, H, r, k, v, w, u)
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
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
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
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                    gw = torch.sum(gw, 0).view(H, C//H)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu)

        def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
            return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################



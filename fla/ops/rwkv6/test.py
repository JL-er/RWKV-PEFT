from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
import os
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

B = 2
T = 4096
C = 4096
HEAD_SIZE = 64

# B = 2
# T = 256
# C = 32
# HEAD_SIZE = 16

H = C // HEAD_SIZE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def val(x):
    return x.detach().float().cpu().numpy()


DEVICE = 'xpu'
DTYPE = torch.bfloat16


def rmsre(pred, target, eps=1e-8):
    return torch.sqrt(torch.mean(torch.square((pred - target) / (target.abs() + eps))))


def print_diff(name, grad1, grad2):
    abs_diff = (grad1 - grad2).abs()
    max_diff = abs_diff.max().item()
    rmsre_value = rmsre(grad1, grad2).item()
    print(f"{name}: Max Abs Diff = {max_diff:.6f}, RMSRE = {rmsre_value:.6f}")


def RUN_FLA_FP32(B, T, C, H, r, k, v, w, u):
    r = r.view(B, T, H, -1).transpose(1, 2).float()
    k = k.view(B, T, H, -1).transpose(1, 2).float()
    v = v.view(B, T, H, -1).transpose(1, 2).float()
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2).float())
    o, _ = chunk_rwkv6(r, k, v, w, u=u.float(), scale=1, initial_state=None, output_final_state=False)
    return o.transpose(1, 2).reshape(B, T, C)


def RUN_FLA_BF16(B, T, C, H, r, k, v, w, u):
    r = r.view(B, T, H, -1).transpose(1, 2).bfloat16()
    k = k.view(B, T, H, -1).transpose(1, 2).bfloat16()
    v = v.view(B, T, H, -1).transpose(1, 2).bfloat16()
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2).bfloat16())
    o, _ = chunk_rwkv6(r, k, v, w, u=u.bfloat16(), scale=1, initial_state=None, output_final_state=False)
    return o.transpose(1, 2).reshape(B, T, C).bfloat16()


def RUN_FLA1_FP32(B, T, C, H, r, k, v, w, u):
    r = r.view(B, T, H, -1).transpose(1, 2).float()
    k = k.view(B, T, H, -1).transpose(1, 2).float()
    v = v.view(B, T, H, -1).transpose(1, 2).float()
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2).float())
    o, _ = fused_recurrent_rwkv6(r, k, v, w, u=u.float(), scale=1, initial_state=None, output_final_state=False)
    return o.transpose(1, 2).reshape(B, T, C)


def RUN_FLA1_BF16(B, T, C, H, r, k, v, w, u):
    r = r.view(B, T, H, -1).transpose(1, 2).bfloat16()
    k = k.view(B, T, H, -1).transpose(1, 2).bfloat16()
    v = v.view(B, T, H, -1).transpose(1, 2).bfloat16()
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2).bfloat16())
    o, _ = fused_recurrent_rwkv6(r, k, v, w, u=u.bfloat16(), scale=1, initial_state=None, output_final_state=False)
    return o.transpose(1, 2).reshape(B, T, C).bfloat16()


######################################################################################################

def LOSS(y):
    return ((y * y) - torch.tanh(y)).sum()


set_seed(42)
with torch.no_grad():
    r = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
    k = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
    v = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
    w = torch.empty(B, T, C, device=DEVICE).uniform_(-8, 1).to(dtype=DTYPE).float()
    u = torch.empty(H, HEAD_SIZE, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()


def clear_grad():
    r.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    w.requires_grad_()
    u.requires_grad_()
    if r.grad is not None:
        r.grad.data.zero_()
    if k.grad is not None:
        k.grad.data.zero_()
    if v.grad is not None:
        v.grad.data.zero_()
    if w.grad is not None:
        w.grad.data.zero_()
    if u.grad is not None:
        u.grad.data.zero_()


clear_grad()

print(f'B={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE}')


# y16 = RUN_FLA1_BF16(B, T, C, H, r, k, v, w, u)
# LOSS(y16).backward()
# gr2 = r.grad.data.clone()
# gk2 = k.grad.data.clone()
# gv2 = v.grad.data.clone()
# gw2 = w.grad.data.clone()
# gu2 = u.grad.data.clone()
# clear_grad()


# yF32 = RUN_FLA_FP32(B, T, C, H, r, k, v, w, u)
# LOSS(yF32).backward()
# gr3 = r.grad.data.clone()
# gk3 = k.grad.data.clone()
# gv3 = v.grad.data.clone()
# gw3 = w.grad.data.clone()
# gu3 = u.grad.data.clone()
# clear_grad()

yF16 = RUN_FLA_BF16(B, T, C, H, r, k, v, w, u)
LOSS(yF16).backward()
gr4 = r.grad.data.clone()
gk4 = k.grad.data.clone()
gv4 = v.grad.data.clone()
gw4 = w.grad.data.clone()
gu4 = u.grad.data.clone()
clear_grad()

y32 = RUN_FLA1_FP32(B, T, C, H, r, k, v, w, u)
LOSS(y32).backward()
gr = r.grad.data.clone()
gk = k.grad.data.clone()
gv = v.grad.data.clone()
gw = w.grad.data.clone()
# print(gw)
gu = u.grad.data.clone()
clear_grad()

# print('fla fused bf16 (fp32 internal)')
# print('y', get_err_ratio(y16, y32))
# print('gr', get_err_ratio(gr2, gr))
# print('gk', get_err_ratio(gk2, gk))
# print('gv', get_err_ratio(gv2, gv))
# print('gw', get_err_ratio(gw2, gw))
# print('gu', get_err_ratio(gu2, gu))

# print('fla chunk fp32')
# print('y', get_err_ratio(yF32, y32))
# print('gr', get_err_ratio(gr3, gr))
# print('gk', get_err_ratio(gk3, gk))
# print('gv', get_err_ratio(gv3, gv))
# print('gw', get_err_ratio(gw3, gw))
# print('gu', get_err_ratio(gu3, gu))

print('fla chunk bf16')
print('y', get_err_ratio(yF16, y32))
print('gr', get_err_ratio(gr4, gr))
print('gk', get_err_ratio(gk4, gk))
print('gv', get_err_ratio(gv4, gv))
print('gw', get_err_ratio(gw4, gw))
print('gu', get_err_ratio(gu4, gu))
print_diff("gw", gw4, gw)
print_diff("gu", gu4, gu)
print_diff("gr", gr4, gr)
print_diff("gk", gk4, gk)
print_diff("gv", gv4, gv)

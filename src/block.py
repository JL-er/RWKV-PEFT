import os
import torch
from .infctx_module import *
import torch.nn as nn
from torch.nn import functional as F

from .rwkv5.rwkv_channel_mix import RWKV_ChannelMix
from .rwkv5.rwkv_time_mix import RWKV_TimeMix_RWKV5
from .rwkv6.rwkv_channel_mix import RWKV_CMix_x060, RWKV_CMix_x060_infctx
from .rwkv6.rwkv_time_mix import RWKV_Tmix_x060, RWKV_Tmix_x060_state, RWKV_Tmix_x060_infctx
from .args_type import TrainingArgs

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


class MishGLU(MyModule):
    def __init__(self, args: TrainingArgs, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))
########################################################################################################


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args: TrainingArgs, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1, args.my_pos_emb, args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb, 1, args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'states':
                    self.att = RWKV_Tmix_x060_state(args, layer_id)
                elif os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.att = RWKV_Tmix_x060_infctx(args, layer_id)
                else:
                    self.att = RWKV_Tmix_x060(args, layer_id)
            else:
                self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
                    self.ffn = RWKV_CMix_x060_infctx(args, layer_id)
                else:
                    self.ffn = RWKV_CMix_x060(args, layer_id)
            else:
                self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
        def forward(self, x, last_state: BlockState, x_emb=None):
            args = self.args
            B, T, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
                if args.my_pos_emb > 0:
                    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                    x = x + pos_emb

            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
                x = x if self.args.dropout == 0 else self.drop0(x)
            else:
                att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
                x = x + att_out
                x = x if self.args.dropout == 0 else self.drop0(x)
            ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
            x = x + ffn_out
            x = x if self.args.dropout == 0 else self.drop1(x)

            if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
                xx = self.tiny_ln(x)
                q = self.tiny_q(xx)[:, :T, :]
                k = self.tiny_k(xx)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
                c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
                x = x + c @ self.tiny_v(x_emb)
            return x, BlockState(att_state, fnn_state)
    else:
        def forward(self, x, x_emb=None):
            args = self.args
            B, T, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
                if args.my_pos_emb > 0:
                    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                    x = x + pos_emb

            if self.args.dropout == 0:
                if self.layer_id == 0 and args.pre_ffn > 0:
                    x = x + self.ffnPre(self.ln1(x))
                else:
                    x = x + self.att(self.ln1(x))
                x = x + self.ffn(self.ln2(x))
            else:
                if self.layer_id == 0 and args.pre_ffn > 0:
                    x = self.drop0(x + self.ffnPre(self.ln1(x)))
                else:
                    x = self.drop0(x + self.att(self.ln1(x)))
                x = self.drop1(x + self.ffn(self.ln2(x)))

            if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
                xx = self.tiny_ln(x)
                q = self.tiny_q(xx)[:, :T, :]
                k = self.tiny_k(xx)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
                c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
                x = x + c @ self.tiny_v(x_emb)
            return x

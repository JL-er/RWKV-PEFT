import os
import torch
from .infctx_module import *
import torch.nn as nn
from torch.nn import functional as F

from .rwkv5.rwkv_channel_mix import RWKV_ChannelMix
from .rwkv5.rwkv_time_mix import RWKV_TimeMix_RWKV5
from .rwkv6.rwkv_channel_mix import RWKV_CMix_x060, RWKV_CMix_x060_infctx
from .rwkv6.rwkv_time_mix import RWKV_Tmix_x060, RWKV_Tmix_x060_state, RWKV_Tmix_x060_infctx
from .rwkv7.Channel_mix import RWKV_CMix_x070
from .rwkv7.Time_mix import RWKV_Tmix_x070

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################
if 'x070' in os.environ["RWKV_MY_TESTING"]:
    class Block(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.ln1 = nn.LayerNorm(args.n_embd)
            self.ln2 = nn.LayerNorm(args.n_embd)

            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(args.n_embd)

            self.att = RWKV_Tmix_x070(args, layer_id)  
            self.ffn = RWKV_CMix_x070(args, layer_id)


        def forward(self, x, v_first):
            if self.layer_id == 0:
                x = self.ln0(x)

            x_attn, v_first = self.att(self.ln1(x), v_first)
            x = x + x_attn

            x = x + self.ffn(self.ln2(x))
            return x, v_first
else:
    class Block(nn.Module):
        def __init__(self, args, layer_id):
            super().__init__()
            self.args = args
            self.layer_id = layer_id

            self.ln1 = nn.LayerNorm(args.n_embd)
            self.ln2 = nn.LayerNorm(args.n_embd)

            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(args.n_embd)
                if args.my_pos_emb > 0:
                    self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                    self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

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
                self.drop0 = nn.Dropout(p = args.dropout)
                self.drop1 = nn.Dropout(p = args.dropout)

        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def forward(self, x, last_state: BlockState, x_emb=None):
                args = self.args
                B, T, C = x.size()
                if self.layer_id == 0:
                    x = self.ln0(x)
                    if args.my_pos_emb > 0:
                        pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                        x = x + pos_emb

                if self.args.dropout == 0:
                    if self.layer_id == 0 and args.pre_ffn > 0:
                        x = x + self.ffnPre(self.ln1(x))
                    else:
                        att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
                        x = x + att_out
                    ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
                    x = x + ffn_out
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
                return x, BlockState(att_state, fnn_state)
        else:
            def forward(self, x, x_emb=None):
                args = self.args
                B, T, C = x.size()
                if self.layer_id == 0:
                    x = self.ln0(x)
                    if args.my_pos_emb > 0:
                        pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
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

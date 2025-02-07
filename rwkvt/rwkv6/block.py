import os
import torch.nn as nn
from .ffn import RWKV_Cmix_v6
from .att import RWKV_Tmix_v6
from rwkvt.infctx_module import BlockState
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v6(args, layer_id)  
        self.ffn = RWKV_Cmix_v6(args, layer_id)


    # def forward(self, x, v_first):
    #     if self.layer_id == 0:
    #         x = self.ln0(x)

    #     x_attn, v_first = self.att(self.ln1(x), v_first)
    #     x = x + x_attn

    #     x = x + self.ffn(self.ln2(x))
    #     return x, v_first
    @property
    def _use_infctx(self):
        """判断是否使用无限上下文模式"""
        return os.environ.get("RWKV_TRAIN_TYPE") == 'infctx'

    def forward(self, *args, **kwargs):
        if self._use_infctx:
            return self.forward_infctx(*args, **kwargs)
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_infctx(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn,  att_state = self.att(self.ln1(x), last_state.time_mix_state)
        x = x + x_attn

        ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
        x = x + ffn_out
        return x, BlockState(att_state, ffn_state)
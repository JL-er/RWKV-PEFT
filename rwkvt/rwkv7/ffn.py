import os
import torch
import torch.nn as nn
from rwkvt.infctx_module import *
from rwkvt.peft.rwkvLinear import make_linear_ffn

def RWKV_Cmix_v7(*args, **kwargs):
    
    if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
        return RWKV_CMix_x070_infctx(*args, **kwargs)
    else:
        return RWKV_CMix_x070(*args, **kwargs)

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = make_linear_ffn(args.n_embd, args.n_embd * 4, bias=False)
        self.value = make_linear_ffn(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    

class RWKV_CMix_x070_infctx(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = make_linear_ffn(args.n_embd, args.n_embd * 4, bias=False)
        self.value = make_linear_ffn(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x,last_state: ChannelMixState):
        #xx = self.time_shift(x) - x
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k),ChannelMixState(x[:, -1])
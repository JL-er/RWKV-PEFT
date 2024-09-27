from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import bitsandbytes as bnb
from argparse import ArgumentParser
from einops import rearrange
parser = ArgumentParser()
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--lora_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)
parser.add_argument("--quant", default="none", type=str)
parser.add_argument("--device", default="cuda", type=str)
args = parser.parse_args()
device= args.device
base_model = args.base_model
lora= args.lora_checkpoint
output= args.output
quant= args.quant

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')

    for k in w_lora.keys():
        w[k] = w_lora[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            gbmm = prefix + '.bone'
            
            # if gbmm in keys:  ##old
            #     w[k] = w[k].to(device=device)
            #     w[gbmm] = w[gbmm].to(device=device)
            #     b,r = w[gbmm].shape

            #     bone = rearrange(w[k], '(a r1) (b r2) -> b a r1 r2', r1 = r, r2 = r)@w[gbmm].reshape(b//r, r, r)+w[gbmm].reshape(b//r, r, r)
            #     w[k] += rearrange(bone, 'b a r1 r2 ->(a r1) (b r2) ')

            #     output_w[k] = w[k].to(device='cpu', copy=True)
            #     del w[k]
            #     del w[gbmm]
            #     continue

            # if gbmm in keys: ### col
            #     w[k] = w[k].to(device=device)
            #     w[gbmm] = w[gbmm].to(device=device)
            #     b,r,_ = w[gbmm].shape
            #     bone = rearrange(w[k], '(a r1) (b r2) -> b a r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
            #     w[k] += rearrange(bone, 'b a r1 r2 ->(a r1) (b r2) ')

            #     output_w[k] = w[k].to(device='cpu', copy=True)
            #     del w[k]
            #     del w[gbmm]
            #     continue

            if gbmm in keys: ### row
                w[k] = w[k].to(device=device)
                w[gbmm] = w[gbmm].to(device=device)
                b,r,_ = w[gbmm].shape
                bone = rearrange(w[k], '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                w[k] += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')

                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[gbmm]
                continue

        if 'bone' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)
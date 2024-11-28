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
                dim = w[gbmm].dim()
                if quant=='4bit':
                    qw,qs = bnb.functional.quantize_4bit(w[k])
                    w[k] = (bnb.functional.dequantize_4bit(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                elif quant=='nf4':
                    qw,qs = bnb.functional.quantize_nf4(w[k])
                    w[k] = (bnb.functional.dequantize_nf4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                elif quant=='fp4':
                    qw,qs = bnb.functional.quantize_fp4(w[k])
                    w[k] = (bnb.functional.dequantize_fp4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                elif quant=='int8':
                    qw,qs = bnb.functional.quantize(w[k])
                    w[k] = (bnb.functional.dequantize(qw,state=qs)).to(dtype=torch.bfloat16)
                if dim==3:
                    b,r,_ = w[gbmm].shape
                    bone = rearrange(w[k], '(a r1) (b r2) -> a b r1 r2', r1 = r, r2 = r)@w[gbmm]+w[gbmm]
                    w[k] += rearrange(bone, 'a b r1 r2 ->(a r1) (b r2) ')
                if dim==2:
                    r,b = w[gbmm].shape
                    
                    in_features = w[k].size(-1)
                    if in_features % r != 0:
                        last_size = in_features % r
                        n_block = in_features // r
                        n_block_size = n_block * r
                        w[k][:, :n_block_size] = (
                            (w[k][:, :n_block_size].reshape(-1, n_block, r).permute(1, 2, 0) + w[gbmm])
                            .permute(2, 0, 1)
                            .reshape(*w[k][:, :n_block_size].shape)
                        )
                        w[k][:, n_block_size:] = (
                            w[k][:, n_block_size:] + (w[gbmm].transpose(0, 1))[:, :last_size]
                        )
                    else:
                        t = w[k].reshape(-1, w[k].size(-1) // r, r).permute(1, 2, 0) + w[gbmm]
                        w[k] = t.permute(2, 0, 1).reshape(*w[k].shape)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[gbmm]
                continue

        if 'bone' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)
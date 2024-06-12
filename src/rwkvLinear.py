import torch, math
import torch.nn as nn
import bitsandbytes as bnb
from torch.nn import functional as F
from torch._lowrank import svd_lowrank
import functools

def rwkv_quantize(quant_type, weight):
    if quant_type=='4bit':
        qweight, qstate= bnb.functional.quantize_4bit((weight.data).to('cuda'))
    elif quant_type=='nf4':
        qweight, qstate= bnb.functional.quantize_nf4((weight.data).to('cuda'))
    elif quant_type=='fp4':
        qweight, qstate= bnb.functional.quantize_fp4((weight.data).to('cuda'))
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize((weight.data).to('cuda'))
    return qweight, qstate


def rwkv_dequantize(quant_type, weight, qstate):
    if quant_type=='4bit':
        deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate)
    elif quant_type=='nf4':
        deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate)
    elif quant_type=='fp4':
        deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate)
    elif quant_type=='int8':
        deweight= bnb.functional.dequantize(weight.data,state=qstate)
    return deweight


        
LORA_CONFIG = {
    "r": 0,
    "alpha": 0,
    "dropout": 0,
    "parts": {"att", "ln", "time", "ffn"},
    "quant": False,
}
class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
            "alpha"], LORA_CONFIG["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.r = r
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.pissa = False
        self.is_quant = False

    def pissa_init(self, svd_niter):

        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A
    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))

    def forward(self, x):

        if self.is_quant:
            if self.pissa:
                return (
                    F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate).to(torch.bfloat16)) + 
                    F.linear(F.linear(x, self.lora_A), self.lora_B))
            return (
                F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate)) + self.scaling *
                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 

        if self.pissa:
            return (
                F.linear(x, self.weight) + 
                F.linear(F.linear(x, self.lora_A), self.lora_B))
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))  
    

class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))
    def forward(self, x):

        if self.is_quant:
            return F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate).to(torch.bfloat16))
        else:
            return F.linear(x, self.weight)
        

@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    elif LORA_CONFIG["quant"]:
        return QuantLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    elif LORA_CONFIG["quant"]:
        return QuantLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)
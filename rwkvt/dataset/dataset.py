########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn.functional as F

import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from .binidx import MMapIndexedDataset
from rwkvt.args_type import TrainingArgs
from rwkv.utils import PIPELINE
from rwkvt.dataset.SFTdataset import sft_dataset
import time

from .mask import generate_mask, create_mask,mask_fn_dict
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")

def get_vocab_size(args: TrainingArgs) -> int:
    train_data = MyDataset(args)
    temp = train_data.vocab_size
    del train_data
    return int(temp)

def get_data_by_l_version(trainer: L.Trainer, args: TrainingArgs):
    if L.__version__[0] == '1':
        train_data = MyDataset(args)
        args.vocab_size = train_data.vocab_size
        train_data.real_epoch = trainer.current_epoch
        train_data.rank = trainer.global_rank
        train_data.world_size = trainer.world_size
        train_data.setup(trainer.global_rank, trainer.world_size, 
                        int(args.devices), args.data_shuffle)
        train_data = DataLoader(train_data, shuffle=args.data_shuffle, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    
    elif L.__version__[0] == '2':
        train_data = MyDataModule(args)
    else:
        raise ValueError(f"Unsupported PyTorch Lightning version: {L.__version__}")
    return train_data

class GlobalIndexManager:
    def __init__(self, rank=0, device_num=1, shuffle=True):
        self.current_idx = 0
        self.rank = rank
        self.device_num = device_num
        self.shuffle = shuffle
        
    def get_next_idx(self, idx_t):
        if self.shuffle:
            idx = idx_t
        else:
            idx = self.current_idx * self.device_num + self.rank 
            self.current_idx += 1
        return idx

class MyDataModule(L.LightningDataModule):
    def __init__(self, args: TrainingArgs):
        super().__init__()
        self.args = args
        self.train_data = None
        
    def setup(self, stage=None):
        self.train_data = MyDataset(self.args)
        self.args.vocab_size = self.train_data.vocab_size
        self.train_data.real_epoch = self.trainer.current_epoch
        self.train_data.rank = self.trainer.global_rank
        self.train_data.world_size = self.trainer.world_size
        self.train_data.setup(self.trainer.global_rank, self.trainer.world_size, 
                              int(self.args.devices), self.args.data_shuffle)
        
    def train_dataloader(self):
        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        return DataLoader(
            self.train_data,
            shuffle=self.args.data_shuffle,
            pin_memory=True,
            batch_size=self.args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )

class MyDataset(Dataset):
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0
        self.index_manager = None
        self.vocab_size = args.vocab_size
        if args.data_type == "sft":
            self.data = sft_dataset(args)
        elif args.data_type == "jsonl":
            import jsonlines
            with jsonlines.open(args.data_file) as file:
                self.data = list(file)

        elif args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")



    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)
    
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        idx = self.index_manager.get_next_idx(idx_t=idx) if self.index_manager else idx
        args = self.args
        rank = self.rank
        epoch = self.real_epoch
        world_size = self.world_size


        if args.data_type == "sft":

            inputs, labels, attn_mask = self.data[0][idx], self.data[1][idx], self.data[2][idx]
            labels= torch.roll(labels, shifts=-1, dims=-1)

            return inputs, labels, attn_mask
        elif args.data_type == "jsonl":
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            ctx = self.data[idx]['text']
            token = torch.tensor(pipeline.encode(ctx))
            token_len = len(token)
            min_len = min(token_len, req_len)
            if req_len < token_len :
                token = token[:req_len]
                pad_len = 0
            else:
                pad_len = req_len - token_len
        
            # dix = F.pad(token, (pad_len, 0), value=0)
            dix = F.pad(token, (0, pad_len), value=0)
            label = F.pad(token, (0, pad_len), value=-100)
            x = dix[:-1]
            y = label[1:]
            # mask = torch.zeros(req_len)
            # mask[pad_len:] = 1

            # mask = torch.zeros(req_len)
            # mask[:min_len] = 1
        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            data = self.data


            if args.data_type == "binidx":
                if args.dataload == 'pad':
                    dix, min_len = data.pad(idx=idx, length=req_len)
                elif args.dataload == 'only':
                    dix = data.only(idx=idx, length=req_len).astype(int)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            dix[min_len:] = -100
            y = torch.tensor(dix[1:], dtype=torch.long)

        mask_fn = mask_fn_dict.get(args.loss_mask)

        if mask_fn!=None:
            t1 = pipeline.encode('User:')
            t2 = pipeline.encode('Assistant:')
            y = mask_fn(dix, t1, t2, min_len)
        return x, y
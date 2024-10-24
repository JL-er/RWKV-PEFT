########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from rwkv.utils import PIPELINE
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
        train_data = DataLoader(train_data, shuffle=args.data_shuffle, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    
    elif L.__version__[0] == '2':
        train_data = MyDataModule(args)
    else:
        raise ValueError(f"Unsupported PyTorch Lightning version: {L.__version__}")
    return train_data

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
        
    def train_dataloader(self):
        # 处理shuffle逻辑
        data_shuffle = True if self.args.data_shuffle == 1 else False
        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        return DataLoader(
            self.train_data,
            shuffle=data_shuffle,
            pin_memory=True,
            batch_size=self.args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            if args.my_pile_version == 1:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
                rank_zero_info(f"Data has {self.data_size} tokens.")
            elif args.my_pile_version == 2:
                data_list = open(args.data_file, "r", encoding='utf-8').read().strip().split('\n')
                data_list = [i.strip().split(' ') for i in data_list]
                self.data = []
                self.data_size = int(data_list[-1][-1])
                rank_zero_info(f"Data has {self.data_size} chunks.")
                for d in data_list:
                    data = MMapIndexedDataset(d[0])
                    data_size = len(data._bin_buffer) // data._index._dtype_size
                    assert (data_size - args.ctx_len) == int(d[1])
                    self.data += [[int(d[-1]), int(d[1]), data]]
                # rank_zero_info(self.data)

            if args.my_qa_mask > 0:
                # self.data_pile = MMapIndexedDataset('/fsx/pile/pile_20B_tokenizer_text_document')
                self.data_pile = MMapIndexedDataset('/fsx/pile_deduped/pile_0.87_deduped_text_document')
                self.data_pile_size = len(self.data_pile._bin_buffer) // self.data._index._dtype_size
            else:
                self.data_pile = None
                self.data_pile_size = 0

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                if args.my_pile_stage != 4:
                    assert MaybeIsPrime(args.magic_prime)
                    assert args.magic_prime % 3 == 2
                    assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # rank_zero_info()
            # for u in unique:
            #     print(u, end=' ')
            # rank_zero_info('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-8") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.rank
        epoch = self.real_epoch
        world_size = self.world_size

        devices = int(args.devices)
        if devices>1:
            idx = idx*devices+rank

        if args.data_type == "uint16":
            i = np.random.randint(0, self.data_size-1)
            dix = self.data[i]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            magic_prime = args.magic_prime
            data = self.data

            if args.my_pile_stage > 0:
                ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                if args.my_qa_mask > 0:
                    ii_orig = ii
                    if ii % 2 == 0:
                        ii = -1
                        data = self.data_pile
                    else:
                        ii = ii // 2
                if data == self.data_pile:
                    i = np.random.randint(0, self.data_pile_size - req_len)
                else:
                    if args.my_pile_stage == 4 or ii < args.my_random_steps:
                        # cheat: pick a random spot in dataset
                        if args.my_pile_version == 1:
                            i = np.random.randint(0, self.data_size - req_len)
                        else:
                            i = np.random.randint(0, self.data_size)
                    else:
                        ii = ii - args.my_random_steps
                        factor = (math.sqrt(5) - 1) / 2
                        factor = int(magic_prime * factor)
                        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                        i = i + args.my_pile_shift
                # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
            else:
                # cheat: pick a random spot in dataset
                i = np.random.randint(0, self.data_size - req_len)

            if args.data_type == "binidx":
                if args.my_pile_version == 1:
                    if args.dataload == 'pad':
                        dix, min_len = data.pad(idx=idx, length=req_len)
                    elif args.dataload == 'only':
                        dix = data.only(idx=idx, length=req_len).astype(int)
                    else:
                        dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                else:
                    # self.data : cutoff, chunk_count, data
                    for j in range(len(data)):
                        if i < data[j][0]:
                            ii = i
                            i = (i - (data[j-1][0] if j > 0 else 0)) % data[j][1]
                            dix = data[j][2].get(idx=0, offset=i, length=req_len).astype(int)
                            # print(ii, j, i)
                            break
            elif args.data_type == "numpy":
                dix = data[i : i + req_len]
            else:
                dix = [self.stoi[s] for s in data[i : i + req_len]]

            if args.my_qa_mask == 1:
                if data == self.data_pile:
                    z = [1] * ctx_len
                else:
                    z = [0] * ctx_len
                    z_sum = 0
                    isGood = False
                    for i in range(3, ctx_len):
                        if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                            isGood = True
                        if dix[i] == 0:
                            isGood = False
                        if isGood:
                            z[i] = 1
                            z_sum += 1
                    if z_sum == 0:
                        z = [1] * ctx_len
                        i = np.random.randint(0, self.data_pile_size - req_len)
                        dix = self.data_pile.get(idx=0, offset=i, length=req_len).astype(int)
                z = torch.tensor(z, dtype=torch.bfloat16)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)

            # if ii_orig < 50:
            #     # if rank == 1:
            #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
            # else:
            #     exit(0)

            if args.my_qa_mask == 1:
                return x, y, z
            if args.loss_mask=='qa':

                t1 = pipeline.encode('User:')
                t2 = pipeline.encode('Assistant:')
                mask = self.create_mask(dix, t1, t2, min_len)
                return x, y, mask
            
            if args.loss_mask=='pad':
                mask = torch.zeros(req_len-1)
                mask[:min_len-1] = 1
                return x, y, mask
            if args.loss_mask=='se':
                t1 = pipeline.encode(args.mask_id['mask0'])
                t2 = pipeline.encode(args.mask_id['mask1'])
                mask = self.generate_mask(dix, t1, t2, min_len)
                return x, y, mask
                

            return x, y
        
    def create_mask(self, seq, token1, token2, min_len):
        # 找到所有特殊标记的索引
        indices1 = []
        for i in range(min_len - len(token1) + 1):
            if np.array_equal(seq[i:i + len(token1)], token1):
                indices1.append(i)
        indices2 = []

        for i in range(min_len - len(token2) + 1):
            if np.array_equal(seq[i:i + len(token2)], token2):
                indices2.append(i)
        mask = torch.zeros(seq.shape)
        #assert len(indices2)!=0 and len(indices1)!=0
        select = 0
        for i in range(min_len):
            if i in indices1:
                select = 0
            elif i in indices2:
                select = 1
            mask[i] = select
        if torch.sum(mask)==0:
            mask[:min_len-1] = 1
        return mask[1:]
    
    def generate_mask(seq, token1, token2, min_len):
        mask = torch.zeros(seq.shape)  # 初始化mask列表，默认全为0
        current_mask_value = 0  # 初始状态下，所有位置的mask值为0

        i = 0
        while i < min_len:
            if seq[i:i+len(token1)] == token1:
                current_mask_value = 0
                for j in range(len(token1)):
                    mask[i + j] = current_mask_value
                i += len(token1)
            elif seq[i:i+len(token2)] == token2:
                current_mask_value = 1
                for j in range(len(token2)):
                    mask[i + j] = current_mask_value
                i += len(token2)
            else:
                mask[i] = current_mask_value
                i += 1

        if torch.sum(mask)==0:
            mask[:min_len-1] = 1
        return mask[1:]


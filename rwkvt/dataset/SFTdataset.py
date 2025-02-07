
import copy
from typing import Optional, Dict, Sequence, List, Literal
import logging
import torch.nn.functional as F

import torch
import torch.distributed
import transformers
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass, field
# from transformers import AutoTokenizer, Rwkv6Config, Rwkv6Model, Rwkv6Tokenizer

#tokenizer = Rwkv6Tokenizer.from_pretrained("RWKV/v6-Finch-1B6-HF")

tokenizer_path = 'RWKV/rwkv-5-world-3b'
IGNORE_INDEX = -100
EOT_TOKEN = "\x17"

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def sft_dataset(script_args):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=script_args.ctx_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    # logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    # logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    raw_train_datasets = load_dataset(script_args.data_file, split=script_args.sft_split)

    # if script_args.local_rank > 0: 
    #     torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.sft_field[0], "response": script_args.sft_field[1]}
    )
    #labels_tensor = torch.tensor(train_dataset['labels'])
    #input_ids_tensor = torch.tensor(train_dataset['input_ids'])
    labels_tensor = [torch.tensor(label) for label in train_dataset['labels']]
    input_ids_tensor = [torch.tensor(input_id) for input_id in train_dataset['input_ids']]
    return (input_ids_tensor, labels_tensor)


# class Data():
#     model_name_or_path: str = "RWKV/rwkv-5-world-3b"
#     data_file :str = "/home/rwkv/JL/data/MetaMathQA"
#     sft_split: str = "train[:5000]"#field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
#     sft_field: List[str] = ["query", "response"]#field(default=["query", "response"], metadata={"help": "Fields of dataset input and output."})
#     ctx_len: int = 100


# if __name__ == "__main__":
#     args = Data()
#     sft_dataset(args)


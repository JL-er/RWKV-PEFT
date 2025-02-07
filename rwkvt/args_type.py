from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class TrainingArgs:
    load_model: str = ""
    wandb: str = ""
    proj_dir: str = "out"
    random_seed: int = -1
    data_file: str = ""
    data_type: str = "utf-8"
    vocab_size: int = 0
    ctx_len: int = 1024
    epoch_steps: int = 1000
    epoch_count: int = 500
    epoch_begin: int = 0
    epoch_save: int = 5
    micro_bsz: int = 12
    n_layer: int = 6
    n_embd: int = 512
    dim_att: int = 0
    dim_ffn: int = 0
    pre_ffn: int = 0
    head_qk: int = 0
    tiny_att_dim: int = 0
    tiny_att_layer: int = -999
    lr_init: float = 6e-4
    lr_final: float = 1e-5
    warmup_steps: int = -1
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-8
    grad_cp: int = 0
    dropout: float = 0
    weight_decay: float = 0
    weight_decay_final: float = -1
    my_pile_version: int = 1
    my_pile_stage: int = 0
    my_pile_shift: int = -1
    my_pile_edecay: int = 0
    layerwise_lr: int = 1
    ds_bucket_mb: int = 200
    my_sample_len: int = 0
    my_ffn_shift: int = 1
    my_att_shift: int = 1
    head_size_a: int = 64
    head_size_divisor: int = 8
    my_pos_emb: int = 0
    load_partial: int = 0
    magic_prime: int = 0
    my_qa_mask: int = 0
    my_random_steps: int = 0
    rwkv_version: str = 'x060'
    my_exit: int = 99999999
    my_exit_tokens: int = 0
    peft: str = "none"
    train_parts: List[str] = field(default_factory=lambda: ["time", "ln"])
    l2warp_sparse: int = 0
    lora_config: Dict = field(default_factory=lambda: json.loads(
        '{"lora_load":"", "lora_r":8, "lora_alpha":32, "lora_dropout":0.01}'))
    pissa_config: Dict = field(default_factory=lambda: json.loads(
        '{"pissa_load":"", "pissa_init":"", "pissa_r":8, "svd_niter":4}'))
    bone_config: Dict = field(default_factory=lambda: json.loads(
        '{"bone_load":"", "bone_r":64}'))
    quant: str = "none"
    dataload: str = "get"
    state_tune: bool = False
    chunk_ctx: int = 512
    fla: bool = False
    fla_npu: bool = False
    train_type: str = "none"
    loss_mask: str = "none"
    mask_id: Dict = field(default_factory=lambda: json.loads(
        '{"mask0":"0", "mask1":"1"}'))
    data_shuffle: int = 0
    optim: str = "none"
    avg_loss: int = 0
    compile: bool = False

    # PyTorch Lightning specific args
    accelerator: str = "gpu"
    strategy: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    precision: str = "fp16"
    accumulate_grad_batches: int = 1

    # train
    my_timestamp: str = field(init=False)
    enable_checkpointing: bool = False
    replace_sampler_ddp: bool = False
    logger: bool = False
    gradient_clip_val: float = 1.0
    num_sanity_val_steps: int = 0
    check_val_every_n_epoch: int = int(1e20)
    log_every_n_steps: int = int(1e20)
    max_epochs: int = -1
    betas: tuple = field(init=False)
    real_bsz: int = field(init=False)
    run_name: str = field(init=False)
    my_img_version: int = field(init=False)
    my_img_size: int = field(init=False)
    my_img_bit: int = field(init=False)
    my_img_clip: int = field(init=False)
    my_img_clip_scale: int = field(init=False)

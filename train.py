########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from lightning_utilities.core.rank_zero import rank_zero_info
    import lightning as pl
    from lightning.pytorch.strategies import SingleDeviceStrategy, FSDPStrategy, DDPStrategy, DeepSpeedStrategy
    from lightning.pytorch.accelerators.accelerator import Accelerator
    import json
    from src.args_type import TrainingArgs
    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float)  # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float)  # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int)  # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x060', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser.add_argument("--peft", default="none", type=str)  # lora pissa bone
    parser.add_argument("--train_parts", default=["time", "ln"], type=list)  # emb , head
    parser.add_argument("--l2warp_sparse", default=0, type=int)

    # LORA
    parser.add_argument("--lora_config", default='{"lora_load":"", "lora_r":8, "lora_alpha":32, "lora_dropout":0.01}', type=json.loads)

    # #LISA
    # parser.add_argument("--lisa_config", default='{"lisa_r":2, "lisa_k":100}', type=json.loads)

    # PISSA
    parser.add_argument("--pissa_config", default='{"pissa_load":"", "pissa_init":"", "pissa_r":8, "svd_niter":4}', type=json.loads)

    # Bone
    parser.add_argument("--bone_config", default='{"bone_load":"", "bone_r":64}', type=json.loads)

    # quant
    parser.add_argument("--quant", default="none", type=str)

    # dataset
    parser.add_argument("--dataload", default="get", type=str)

    # state tuning
    parser.add_argument("--state_tune", action="store_true")

    parser.add_argument("--chunk_ctx", default=512, type=int)
    # fla
    parser.add_argument("--fla", action="store_true")
    parser.add_argument("--train_type", default="none", type=str)

    # loss_mask
    parser.add_argument("--loss_mask", default="none", type=str)  # pad qa se
    parser.add_argument("--mask_id", default='{"mask0":"0", "mask1":"1"}', type=json.loads)
    parser.add_argument("--data_shuffle", default=1, type=int)

    # new optim
    parser.add_argument("--optim", default="none", type=str)

    # acc_grad_batchs
    parser.add_argument("--avg_loss", default=0, type=int)

    parser.add_argument("--compile", action="store_true")

    if pl.__version__[0] == '2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    args = TrainingArgs(**vars(parser.parse_args()))

    ########################################################################################################

    import os
    import warnings
    import math
    import datetime
    import sys
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    def _args_check():
        assert args.precision in ["fp32", "tf32", "fp16", "bf16-true", "bf16"]
        assert args.accelerator in ["gpu", "xpu", "musa"]
        assert args.strategy in ["auto", "single-device", "fsdp", "ddp", "deepspeed", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"]
        assert args.train_type in ['none', 'state', 'infctx', 'finetune']
        assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]
        if "32" in args.precision:
            args.precision = "32"
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        elif args.precision == "fp16":
            args.precision = "16-mixed"
        elif args.precision == "bf16":
            args.precision = "bf16-mixed"
        else:
            args.precision = "bf16-true"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        if "32"  not in args.precision:
            if args.accelerator == "gpu":
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            elif args.accelerator == "musa":
                import torch_musa
                torch.backends.mudnn.allow_tf32 = True

    deepspeed_version = None
    def _set_env_var():
        global deepspeed_version
        if "deepspeed" in args.strategy:
            import deepspeed
            deepspeed_version = deepspeed.__version__
            os.environ["USE_DEEPSPEED"] = "1"
        
        if args.optim == 'adam_mini':
            os.environ["RWKV_OPTIM"] = 'adam_mini'
        os.environ["RWKV_MY_TESTING"] = args.my_testing
        os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
        # state tuning
        os.environ["RWKV_TRAIN_TYPE"] = ''
        
        if args.train_type == 'state':
            os.environ["RWKV_TRAIN_TYPE"] = 'states'
        elif args.train_type == 'infctx':
            os.environ["RWKV_TRAIN_TYPE"] = 'infctx'

        os.environ["WKV"] = 'fla' if args.fla else ''
        if args.fla:
            os.system('pip uninstall fla -y')
            os.system('pip install --upgrade rwkv-fla')

        os.environ["L2WRAP_SPARSE"] = str(args.l2warp_sparse)

        if args.precision == "32":
            os.environ["RWKV_FLOAT_MODE"] = "32"
        elif args.precision == "16-mixed":
            os.environ["RWKV_FLOAT_MODE"] = "fp16"
        elif isinstance(args.precision, str) and "bf16" in args.precision:
            os.environ["RWKV_FLOAT_MODE"] = "bf16"

        if args.precision == "32":
            for i in range(10):
                rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
        if args.precision == "fp16":
            rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

        os.environ["RWKV_JIT_ON"] = "0"
        if "deepspeed_stage_3" in args.strategy:
            os.environ["RWKV_JIT_ON"] = "0"

        if args.quant != 'none':
            os.environ["RWKV_QUANT"] = 1

        if args.optim == 'adam_mini':
            os.environ["RWKV_OPTIM"] = 'adam_mini'


    if args.random_seed >= 0:
        rank_zero_info(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)
    
    _args_check()
    _set_env_var()

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    warnings.filterwarnings("ignore", "*error: operation scheduled before its operands*")

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    if args.dataload != 'get':
        args.max_epochs = args.epoch_count
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime

        if args.my_pile_shift < 0:
            args.my_pile_shift = 0

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak
        if args.my_qa_mask == 2:
            args.epoch_count = 2 * args.magic_prime // 40320
        else:
            args.epoch_count = args.magic_prime // 40320

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        # if args.my_pile_stage == 2:
        #     assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    if args.my_pile_stage == 2:
                        args.warmup_steps = 10
                    else:
                        args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len

    rank_zero_info(
        f"""
############################################################################
#
# RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")


    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")




    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    if args.peft == 'lora':
        from src.rwkvLinear import LORA_CONFIG
        assert args.lora_config['lora_r'] > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.lora_config['lora_r']
        LORA_CONFIG["alpha"] = args.lora_config['lora_alpha']
        LORA_CONFIG["dropout"] = args.lora_config['lora_dropout']
        # LORA_CONFIG["parts"] = set(str(args.lora_config['lora_parts']).split(','))
    if args.peft == 'pissa':
        assert args.pissa_config['pissa_r'] > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.pissa_config['pissa_r']
        # LORA_CONFIG["parts"] = set(str(args.pissa_config['pissa_parts']).split(','))
    if args.quant != 'none':
        LORA_CONFIG["quant"] = True
    if args.peft == 'bone':
        from src.rwkvLinear import BONE_CONFIG
        BONE_CONFIG["r"] = args.bone_config['bone_r']

    model = RWKV(args)
    print(model)
    freeze = False

    if args.train_type == 'state':
        args.state_tune = True

    if args.train_type == 'state' or args.state_tune:
        model.requires_grad_(False)
        freeze = True
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'state' in pname:
                    param.requires_grad = True
            break
    if args.peft != 'none':
        model.requires_grad_(False)
        freeze = True
        if len(args.load_model) == 0:
            for name, module in model.named_modules():
                if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'emb.weight' == pname:
                            print(f'  EMB additionally training module {pname}')
                            param.requires_grad = True
                if any(n.startswith("head.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'head.weight' == pname:
                            print(f'  head additionally training module {pname}')
                            param.requires_grad = True
                if 'ln' in name:
                    print(f'  LoRA additionally training module {name}')
                    for param in module.parameters():
                        param.requires_grad = True
                break

        for name, module in model.named_modules():  # part train
            for pname, param in module.named_parameters():
                for part in args.train_parts:
                    if part in pname:
                        print(f'  Parts additionally training module {name}')
                        param.requires_grad = True
            break
        # if args.peft=='lisa':
        #     import re
        #     select_layers = np.random.choice(range(args.n_layer), args.lisa_r, replace=False)
        #     for name, module in model.named_modules():
        #         for pname, param in module.named_parameters():
        #             if 'emb' in pname or 'head' in pname or '.ln' in pname or 'time' in pname :
        #                 param.requires_grad = True
        #             match = re.search(r'\d+', pname)
        #             if match:
        #                 number = int(match.group())
        #                 if number in select_layers:
        #                     param.requires_grad  = True
        #         break
        if args.peft == 'lora' or args.peft == 'pissa':
            for name, module in model.named_modules():
                if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                    print(f'  LoRA additionally training module {name}')
                    for pname, param in module.named_parameters():
                        param.requires_grad = 'lora_' in pname
        if args.peft == 'bone':
            for name, module in model.named_modules():
                for pname, param in module.named_parameters():
                    if 'bone' in pname:
                        print(f'  Bone additionally training parameter {pname}')
                        param.requires_grad = True
                break

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    model.load_state_dict(torch.load(args.load_model, map_location="cpu"), strict=(not freeze))

    # Load peft checkpoint
    # multi-GPU training
    if args.peft == 'lora':
        if os.path.isfile(args.lora_config['lora_load']):
            model.load_state_dict(torch.load(args.lora_config['lora_load'], map_location="cpu"),
                                  strict=False)
    elif args.peft == 'pissa':
        if int(args.devices) == 1 and os.path.isfile(f'{args.proj_dir}/init_pissa.pth'):
            assert os.path.isfile(f'{args.proj_dir}/init_pissa.pth') == False
        if os.path.isfile(f'{args.proj_dir}/init_pissa.pth') and int(args.devices) > 1 and args.pissa_config['pissa_load'] == "":
            pissa_init = torch.load(f'{args.proj_dir}/init_pissa.pth', map_location="cpu")
            rank_zero_info(f"########## Load Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

        if args.pissa_config['pissa_load'] == "" and not os.path.isfile(f'{args.proj_dir}/init_pissa.pth'):
            init_dict = {}
            rank_zero_info(f"########## Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                    m.pissa_init(args.pissa_config['svd_niter'])
                    init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                    init_dict[f'{name}.init_lora_B'] = m.lora_B.data
            torch.save(init_dict, f'{args.proj_dir}/init_pissa.pth')
        if os.path.isfile(args.pissa_config['pissa_load']):
            model.load_state_dict(torch.load(args.pissa_config['pissa_load'], map_location="cpu"),
                                  strict=False)
            pissa_init = torch.load(args.pissa_config['pissa_init'], map_location="cpu")
            rank_zero_info(f"########## Load PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

    if args.quant != 'none':
        rank_zero_info(f"########## Quant... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                m.quant(args.quant)

    
    def _get_strategy(strategy: str, devices: int, accelerator: Accelerator):
        if strategy == "auto":
            if devices == 1:
                return SingleDeviceStrategy(device=devices)
            else:
                return DDPStrategy(parallel_devices=accelerator.get_parallel_devices(devices))
        elif strategy == "single-device":
            return SingleDeviceStrategy(device=devices)
        elif strategy == "fsdp":
            return FSDPStrategy(parallel_devices=accelerator.get_parallel_devices(devices))
        elif strategy == "ddp":
            return DDPStrategy(parallel_devices=accelerator.get_parallel_devices(devices))
        elif "deepspeed" in strategy:
            def get_deepspeed_config(strategy: str, args: TrainingArgs):
                base_config = {
                    "stage": 2,  # 默认值
                    "offload_optimizer": False,
                    "offload_parameters": False,
                    "remote_device": None,
                    "offload_params_device": None,
                    "offload_optimizer_device": None,
                    "allgather_bucket_size": args.ds_bucket_mb * 1000 * 1000,
                    "reduce_bucket_size": args.ds_bucket_mb * 1000 * 1000
                }

                if strategy == "deepspeed":
                    return base_config
                
                parts = strategy.split("_")
                if "stage" in parts:
                    stage_index = parts.index("stage")
                    base_config["stage"] = int(parts[stage_index + 1])
                
                if "offload" in parts:
                    base_config["offload_optimizer"] = True
                    if base_config["stage"] == 3:
                        base_config["offload_parameters"] = True
                
                if "nvme" in parts:
                    base_config["remote_device"] = "nvme"
                    base_config["offload_params_device"] = "nvme"
                    base_config["offload_optimizer_device"] = "nvme"
                
                return base_config

            config = get_deepspeed_config(strategy, args)
            return DeepSpeedStrategy(
                parallel_devices=accelerator.get_parallel_devices(devices),
                **config
            )
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    if args.accelerator.lower() == "gpu":
        actual_acc = args.accelerator  # work for NV, AMD, 沐曦
        actual_strategy = _get_strategy(args.strategy, "cuda", accelerator=actual_acc)
    elif args.accelerator.lower() == "xpu":
        from devices.xpu import XPUAccelerator
        actual_acc = XPUAccelerator()  # work for Intel
        actual_strategy = _get_strategy(args.strategy, "xpu", accelerator=actual_acc)
    elif args.accelerator.lower() == "musa":
        from devices.musa import MUSAAccelerator  # work for Morethreads
        actual_acc = MUSAAccelerator()
        actual_strategy = _get_strategy(args.strategy, "musa", accelerator=actual_acc)
    else:
        raise ValueError(f"Unknown accelerator {args.accelerator}")

    if pl.__version__[0] == '2':
        trainer = Trainer(accelerator=actual_acc, strategy=actual_strategy, devices=args.devices, num_nodes=args.num_nodes, precision=args.precision,
                          logger=args.logger, callbacks=[train_callback(args)], max_epochs=args.max_epochs, check_val_every_n_epoch=args.check_val_every_n_epoch, num_sanity_val_steps=args.num_sanity_val_steps,
                          log_every_n_steps=args.log_every_n_steps, enable_checkpointing=args.enable_checkpointing, accumulate_grad_batches=args.accumulate_grad_batches, gradient_clip_val=args.gradient_clip_val)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
        )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    train_data.real_epoch = trainer.current_epoch
    train_data.rank = trainer.global_rank
    train_data.world_size = trainer.world_size

    data_shuffle = True if args.data_shuffle == 1 else False
    if int(args.devices) > 1:
        data_shuffle = False
    if args.epoch_count > 1:
        data_shuffle = True

    data_loader = DataLoader(train_data, shuffle=args.data_shuffle, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    if args.compile:
        model = torch.compile(model)

    trainer.fit(model, data_loader)

import os
import torch
from src.model import RWKV
from src.args_type import TrainingArgs
from lightning_utilities.core.rank_zero import rank_zero_info
from src.trainer import generate_init_weight
from src.rwkvLinear import LORA_CONFIG

def load_peft_model(args: TrainingArgs):
    freeze = False
    if args.peft == 'lora':
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
                            print(
                                f'  EMB additionally training module {pname}')
                            param.requires_grad = True
                if any(n.startswith("head.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'head.weight' == pname:
                            print(
                                f'  head additionally training module {pname}')
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
                        print(
                            f'  Bone additionally training parameter {pname}')
                        param.requires_grad = True
                break

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name
    else:
        rank_zero_info(f"########## Loading {args.load_model}... ##########")
        model.load_state_dict(torch.load(
            args.load_model, map_location="cpu", weights_only=True), strict=(not freeze))

    # Load peft checkpoint
    # multi-GPU training
    if args.peft == 'bone':
        if os.path.isfile(args.bone_config['bone_load']):
            model.load_state_dict(torch.load(args.bone_config['bone_load'], map_location="cpu", weights_only=True),
                                  strict=False)
    elif args.peft == 'lora':
        if os.path.isfile(args.lora_config['lora_load']):
            model.load_state_dict(torch.load(args.lora_config['lora_load'], map_location="cpu", weights_only=True),
                                  strict=False)
    elif args.peft == 'pissa':
        if int(args.devices) == 1 and os.path.isfile(f'{args.proj_dir}/init_pissa.pth'):
            assert os.path.isfile(f'{args.proj_dir}/init_pissa.pth') == False
        if os.path.isfile(f'{args.proj_dir}/init_pissa.pth') and int(args.devices) > 1 and args.pissa_config['pissa_load'] == "":
            pissa_init = torch.load(
                f'{args.proj_dir}/init_pissa.pth', map_location="cpu", weights_only=True)
            rank_zero_info(f"########## Load Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(
                        pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

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
            model.load_state_dict(torch.load(args.pissa_config['pissa_load'], map_location="cpu", weights_only=True),
                                  strict=False)
            pissa_init = torch.load(
                args.pissa_config['pissa_init'], map_location="cpu", weights_only=True)
            rank_zero_info(f"########## Load PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(
                        pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

    if args.quant != 'none':
        rank_zero_info(f"########## Quant... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                m.quant(args.quant)

    return args, model
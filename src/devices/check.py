from src.args_type import TrainingArgs
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.strategies import SingleDeviceStrategy, FSDPStrategy, DDPStrategy, DeepSpeedStrategy


def get_strategy(args: TrainingArgs, devices: int, accelerator: Accelerator):
    strategy = args.strategy.lower()
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

def get_accelerator(args: TrainingArgs):
    if args.accelerator.lower() == "gpu":
        actual_acc = args.accelerator  # work for NV, AMD, 沐曦
    elif args.accelerator.lower() == "xpu":
        from src.devices.xpu import XPUAccelerator
        actual_acc = XPUAccelerator()  # work for Intel
    elif args.accelerator.lower() == "musa":
        from src.devices.musa import MUSAAccelerator  # work for Morethreads
        actual_acc = MUSAAccelerator()
    elif args.accelerator.lower() == "npu":
        from src.devices.npu import NPUAccelerator
        actual_acc = NPUAccelerator()
    else:
        raise ValueError(f"Unknown accelerator {args.accelerator}")
    return actual_acc
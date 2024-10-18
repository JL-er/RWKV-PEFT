import torch
import torch_npu
from functools import lru_cache
from typing import Any, Dict, Union
from pytorch_lightning.accelerators.accelerator import Accelerator

from typing_extensions import override


class NPUAccelerator(Accelerator):
    """Accelerator for HUAWEI NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type NPU.
        """
        if device.type != "npu":
            raise ValueError(f"Device should be of type 'npu', got '{device.type}' instead.")
        if device.index is None:
            device = torch.device("npu", 0)
        torch.npu.set_device(device.index)

    @override
    def teardown(self) -> None:
        torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        return [torch.device("npu", i) for i in range(torch.npu.device_count())]

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        if type(devices) is int:
            return [torch.device("npu", i) for i in range(devices)]
        elif type(devices) is list:
            try:
                return [torch.device("npu", i) for i in devices]
            except BaseException:
                return devices
        elif type(devices) is str and devices == "auto":
            return [torch.device("npu", i) for i in range(torch.npu.device_count())]
        elif type(devices) is str and devices == "npu":
            return [torch.device("npu", i) for i in range(torch.npu.device_count())]

    @staticmethod
    @override
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.npu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.npu.is_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry):
        accelerator_registry.register(
            "npu",
            cls,
            description=f"NPU Accelerator - optimized for large-scale machine learning.",
        )

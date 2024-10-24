import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    assert torch.xpu.is_available(), "Intel Extension for PyTorch is not installed. Please install it from source."
from functools import lru_cache
from typing import Any, Dict, Union
from lightning.pytorch.accelerators.accelerator import Accelerator

from typing_extensions import override


class XPUAccelerator(Accelerator):
    """Accelerator for INTEL XPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type XPU.
        """
        if device.type != "xpu":
            raise ValueError(f"Device should be of type 'xpu', got '{device.type}' instead.")
        if device.index is None:
            device = torch.device("xpu", 0)
        torch.xpu.set_device(device.index)

    @override
    def teardown(self) -> None:
        torch.xpu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        return [torch.device("xpu", i) for i in range(torch.xpu.device_count())]

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        if type(devices) is int:
            return [torch.device("xpu", i) for i in range(devices)]
        elif type(devices) is list:
            try:
                return [torch.device("xpu", i) for i in devices]
            except BaseException:
                return devices
        elif type(devices) is str and devices == "auto":
            return [torch.device("xpu", i) for i in range(torch.xpu.device_count())]
        elif type(devices) is str and devices == "xpu":
            return [torch.device("xpu", i) for i in range(torch.xpu.device_count())]

    @staticmethod
    @override
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.xpu.is_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry):
        accelerator_registry.register(
            "xpu",
            cls,
            description=f"XPU Accelerator - optimized for large-scale machine learning.",
        )

    @staticmethod
    @override
    def get_device() -> str:
        return "xpu"

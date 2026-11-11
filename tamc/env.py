"""Device selection and environment reporting.

Works on three places without changes:
  - Colab / RTX 3050  -> cuda
  - M1 iMac / M4 Air   -> mps
  - this Windows laptop -> cpu
"""

import platform
import sys

import torch


def pick_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Pick a safe dtype for the device.

    T4 (Colab free) and RTX 3050 have no bf16 support, so fp16 is used on cuda.
    MPS runs fp16 fine. CPU must stay fp32 - fp16 on CPU is slow and buggy.
    """
    if device == "cpu":
        return torch.float32
    return torch.float16


def report() -> dict:
    """Print and return everything needed to reproduce a run."""
    device = pick_device()
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "dtype": str(pick_dtype(device)),
    }

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda"] = torch.version.cuda
        info["bf16_supported"] = torch.cuda.is_bf16_supported()

    width = max(len(k) for k in info)
    print("=" * 52)
    print("ENVIRONMENT")
    print("=" * 52)
    for key, value in info.items():
        print(f"  {key:<{width}} : {value}")
    print("=" * 52)
    return info


if __name__ == "__main__":
    report()

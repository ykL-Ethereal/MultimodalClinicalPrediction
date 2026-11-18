"""Shared CUDA selection and precision policy; no allocation at import time."""
from contextlib import nullcontext

import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {index}")
        return torch.device("cuda", index)
    return device


def resolve_precision(device, precision="auto"):
    if precision not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("precision must be auto, float32, float16 or bfloat16")
    if device.type != "cuda" or precision == "float32":
        return "float32"
    with torch.cuda.device(device):
        bf16 = torch.cuda.is_bf16_supported()
    if precision == "bfloat16" and not bf16:
        raise ValueError(f"{device} does not support bfloat16")
    return ("bfloat16" if bf16 else "float16") if precision == "auto" else precision


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=getattr(torch, precision))

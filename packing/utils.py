import torch
from xformers.ops.fmha.attn_bias import AttentionBias


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(value, device) for value in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, int):
        return data
    elif isinstance(data, AttentionBias):
        return data
    elif data is None:
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

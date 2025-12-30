import torch
from dataclasses import fields
from typing import Optional

from dataset.data_collate import BatchInput


def place_data(batch_data: BatchInput, device: str = 'cuda', dtype: Optional[torch.dtype] = None):
    """
    Move batch data to the specified device and optionally convert dtype.
    Uses dataclasses.fields() to elegantly scan all fields and move tensors.
    
    Args:
        batch_data: Collated batch data (BatchInput dataclass)
        device: Target device ('cuda', 'cpu', or torch.device object)
        dtype: Optional target dtype (e.g., torch.bfloat16, torch.float32)
    
    Returns:
        batch_data with all tensors moved to the specified device and dtype
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    for field in fields(batch_data):
        value = getattr(batch_data, field.name)
        if isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device)
            setattr(batch_data, field.name, value)
    
    return batch_data

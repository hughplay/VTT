import numpy as np
import torch


def tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return [x]


def mask_select(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select tensor elements where mask is True.

    Args:
        tensor: tensor of shape (B, ?)
        mask: tensor of shape (B)

    Returns:
        tensor of shape (M, ?), where M is the number of True elements in mask
    """
    return tensor[mask]


def mask_unselect(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Unselect tensor elements where mask is True.

    Args:
        tensor: tensor of shape (M, ?)
        mask: tensor of shape (B)

    Returns:
        tensor of shape (B, ?), where False positions are filled with 0
    """
    tensor_size = list(tensor.size())
    assert tensor_size[0] == mask.sum().item()
    tensor_size[0] = mask.size(0)
    full_tensor = torch.zeros(
        tensor_size, dtype=tensor.dtype, device=tensor.device
    )
    full_tensor[mask] = tensor
    return full_tensor

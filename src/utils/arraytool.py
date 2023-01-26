import numpy as np
import torch


def tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return [x]

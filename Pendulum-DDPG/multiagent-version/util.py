import torch

def to_tensor(ndarray, device='cpu', requires_grad=False, dtype=torch.float):
    return torch.tensor(ndarray, dtype = dtype, device = device, requires_grad=requires_grad)
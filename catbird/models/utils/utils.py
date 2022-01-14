import torch
import torch.nn as nn


def freeze_params(model: nn.Module) -> None:
    """Disable weight updates on given model.

    Args:
        model (nn.Module): model
    """
    for par in model.parameters():
        par.requires_grad = False


def one_hot(t, c):
    return torch.zeros(*t.size(), c, device=t.device).scatter_(
        -1, t.unsqueeze(-1).data.long(), 1
    )


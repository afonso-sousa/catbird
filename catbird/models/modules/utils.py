from torch import nn


def freeze_params(model: nn.Module) -> None:
    """Disable weight updates on given model.

    Args:
        model (nn.Module): model
    """
    for par in model.parameters():
        par.requires_grad = False

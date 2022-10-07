from ignite.contrib.handlers import PiecewiseLinear
from torch import optim


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    peak_lr: float,
    num_warmup_epochs: int,
    num_epochs: int,
    epoch_length: int,
) -> PiecewiseLinear:
    milestones_values = [
        (0, 0.0),
        (epoch_length * num_warmup_epochs, peak_lr),
        (epoch_length * num_epochs - 1, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )

    return lr_scheduler

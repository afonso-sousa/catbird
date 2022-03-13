from ignite.contrib.handlers import ParamGroupScheduler, PiecewiseLinear


def build_lr_scheduler(
    optimizer, peak_lr, num_warmup_epochs, num_epochs, epoch_length
):
    milestones_values = [
        (0, 0.0),
        (epoch_length * num_warmup_epochs, peak_lr),
        (epoch_length * num_epochs - 1, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values)

    return lr_scheduler

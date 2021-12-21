"""Method to initialize a pretrained T5 conditional generation model."""

from typing import Tuple

import ignite.distributed as idist
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
from transformers import T5ForConditionalGeneration


def __freeze_params(model: nn.Module) -> None:
    """Disable weight updates on given model.

    Args:
        model (nn.Module): model
    """
    for par in model.parameters():
        par.requires_grad = False


def initialize(cfg: Config) -> Tuple[nn.Module, optim.Optimizer]:
    """Initialize T5 conditional generator based on the given configurations.

    Args:
        cfg (Config): configuration file

    Returns:
        Tuple[nn.Module, optim.Optimizer]: model and optimizer
    """
    model = T5ForConditionalGeneration.from_pretrained(cfg.model.name)
    lr = cfg.train.learning_rate * idist.get_world_size()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd_entry in n for nd_entry in no_decay)
            ],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd_entry in n for nd_entry in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if cfg.model.freeze_encoder:
        __freeze_params(model.get_encoder())

    model = idist.auto_model(model)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = idist.auto_optim(optimizer)

    return model, optimizer

"""Factory to build models."""
from importlib import import_module
from typing import Tuple

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger


def build_generator(cfg: Config) -> Tuple[nn.Module, optim.Optimizer]:
    """Abstraction to build models based on the given configurations.

    Args:
        cfg (Config): configuration file

    Raises:
        NameError: Exception raised if generator name in config does not match any of the available options

    Returns:
        nn.Module: selected model based on configurations
    """
    logger = setup_logger(name="Model", distributed_rank=idist.get_rank())
    logger.info(f"Loading {cfg.model.name} dataset")

    model_name = cfg.model.name.lower().split("-")[0]
    module = import_module(f"catbird.models.{model_name}")

    model, optimizer = getattr(module, "initialize")(cfg)

    if cfg.resume_from:
        checkpoint = torch.load(cfg.resume_from)
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    return model, optimizer

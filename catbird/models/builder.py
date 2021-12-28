"""Factory to build models."""

import sys
from importlib import import_module
from typing import Tuple

import ignite.distributed as idist
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
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
    try:
        model_name = cfg.model.name.lower().split("-")[0]
        module = import_module(f"catbird.models.{model_name}")
    except ModuleNotFoundError:
        print(
            f"The generator name '{cfg.model.name}' does not match any of our currently available options."
        )
        sys.exit(1)

    return getattr(module, "initialize")(cfg)


if __name__ == "__main__":
    from catbird.core import Config

    cfg = Config.fromfile("configs/edl_quora.yaml")
    cfg.embedding_length = 10000
    cfg.pad_token_id = 0
    cfg = Config(dict(model=dict(name="testing")))  # check if it throws error
    model, optimizer = build_generator(cfg)

    print(model)

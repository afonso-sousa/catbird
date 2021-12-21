"""Factory to build models."""

from typing import Tuple

import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore

from .t5 import initialize


def build_generator(cfg: Config) -> Tuple[nn.Module, optim.Optimizer]:
    """Abstraction to build models based on the given configurations.

    Args:
        cfg (Config): configuration file

    Raises:
        NameError: Exception raised if generator name in config does not match any of the available options

    Returns:
        nn.Module: selected model based on configurations
    """
    if cfg.model.name.lower() == "t5-small":
        return initialize(cfg)
    else:
        raise NameError(
            "The generator name does not match any of our currently available options."
        )

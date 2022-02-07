"""Factory to build models."""
from importlib import import_module

import ignite.distributed as idist
import torch
from torch import nn
from catbird.core import Config  # type: ignore
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from catbird.core import build_from_cfg
from .utils import freeze_params

from .registry import GENERATORS, ENCODERS, DECODERS

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_generator(cfg: Config) -> nn.Module:
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

    # HARDCODED OVERWRITE
    if isinstance(cfg.model, dict):
        cfg.model.encoder.vocabulary_size = cfg.embedding_length
        cfg.model.encoder.pad_token_id = cfg.pad_token_id
        cfg.model.decoder.vocabulary_size = cfg.embedding_length
        cfg.model.decoder.pad_token_id = cfg.pad_token_id
        model = build_generator2(cfg.model)
    else:
        model_name = cfg.model.name.lower().split("-")[0]
        module = import_module(f"catbird.models.{model_name}")
        
        # module_name = filename.stem
        # if "." in module_name:
        #     raise ValueError("Dots are not allowed in config file path.")
        # config_dir = Path(filename).parent
        # sys.path.insert(0, config_dir.as_posix())
        # mod = import_module(module_name)
        # sys.path.pop(0) 

        model = getattr(module, "initialize")(cfg)

    if cfg.resume_from:
        checkpoint = torch.load(cfg.resume_from)
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    return model


def build_generator2(cfg):
    model = build(cfg, GENERATORS)
    if cfg.freeze_encoder:
        freeze_params(model.get_encoder())

    return idist.auto_model(model)


def build_encoder(cfg):
    return build(cfg, ENCODERS)


def build_decoder(cfg):
    return build(cfg, DECODERS)
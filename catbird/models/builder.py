"""Factory to build models."""
from importlib import import_module

import ignite.distributed as idist
import torch
from catbird.core import Config  # type: ignore
from catbird.core import build_from_cfg
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from torch import nn

from .registry import (DECODERS, DISCRIMINATORS, ENCODERS, GENERATORS,
                       GRAPH_ENCODERS)
from .modules import freeze_params


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_generator_model(cfg: Config) -> nn.Module:
    """Abstraction to build models based on the given configurations.

    Args:
        cfg (Config): configuration file

    Raises:
        NameError: Exception raised if generator name in config does not match any of the available options

    Returns:
        nn.Module: selected model based on configurations
    """
    logger = setup_logger(name="Model", distributed_rank=idist.get_rank())
    logger.info(f"Loading {cfg.model.type} model")

    if isinstance(cfg.model, dict) and "type" in cfg.model:
        if cfg.model.type == "HuggingFaceWrapper":
            cfg.model.vocab_size = cfg.embedding_length
        else:
            for key in cfg.model:
                if key == "type":
                    continue
                cfg.model[key].vocab_size = cfg.embedding_length
                cfg.model[key].pad_token_id = cfg.pad_token_id
        cfg.model.pad_token_id = cfg.pad_token_id
        cfg.model.eos_token_id = cfg.eos_token_id
        cfg.model.decoder_start_token_id = cfg.decoder_start_token_id
        model = build_generator(cfg.model)

    if cfg.resume_from:
        checkpoint = torch.load(cfg.resume_from)
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    logger.info(model)

    return model


def build_generator(cfg):
    model = build(cfg, GENERATORS)
    if cfg.freeze_encoder:
        freeze_params(model.get_encoder())    

    return idist.auto_model(model)


def build_encoder(cfg):
    return build(cfg, ENCODERS)


def build_graph_encoder(cfg):
    return build(cfg, GRAPH_ENCODERS)


def build_decoder(cfg):
    return build(cfg, DECODERS)


def build_discriminator(cfg):
    return build(cfg, DISCRIMINATORS)

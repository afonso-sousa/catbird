"""Main methods for model training and evaluation."""

from importlib import import_module
from logging import Logger
from typing import Optional

import ignite.distributed as idist
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
from ignite.contrib.engines import common
from ignite.engine import Engine
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


def create_trainer(
    cfg: Config,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_sampler: Optional[DistributedSampler],
    logger: Logger,
) -> Engine:
    """Create a training step for the given model.

    Args:
        cfg (Config): configuration file.
        model (nn.Module): model to train.
        optimizer (optim.Optimizer): train optimizer.
        train_sampler (Union[Sampler, Iterable]): defines the strategy to draw
            samples from the dataset.
        logger ([type]): object to log progress to the CLI.

    Returns:
        Engine: Object to run the defined train step over each batch of a dataset.
    """
    device = idist.device()
    scaler = GradScaler(enabled=cfg.train.with_amp)

    model_name = cfg.model.name.lower().split("-")[0]
    module = import_module(f"catbird.models.{model_name}")
    train_step = getattr(module, "train_step")(cfg, model, optimizer, device, scaler)

    trainer = Engine(train_step)
    trainer.logger = logger

    metric_names = ["batch loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        output_names=metric_names,
        clear_cuda_cache=False,
        with_pbars=True,
    )
    return trainer


def create_evaluator(
    cfg: Config,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    metrics: dict,
    logger: Logger,
) -> Engine:
    """Create an evaluation step for the given model.

    Args:
        cfg (Config): configuration file.
        model (nn.Module): model to evaluate.
        tokenizer (AutoTokenizer): tokenizer.
        metrics (dict): dictionary of metrics to evaluate the model.
        logger (Logger): object to log progress to the CLI.

    Returns:
        Engine: Object to run the defined evaluation step over each batch of a dataset.
    """
    device = idist.device()

    model_name = cfg.model.name.lower().split("-")[0]
    module = import_module(f"catbird.models.{model_name}")
    evaluate_step = getattr(module, "evaluate_step")(
        cfg, model, tokenizer, device, logger
    )

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


if __name__ == "__main__":
    import logging

    from catbird.core import Config
    from catbird.datasets import build_dataset, get_dataloaders
    from catbird.models import build_generator
    from catbird.tokenizers import build_tokenizer

    cfg = Config.fromfile("configs/edl_quora.yaml")

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)

    datasets = build_dataset(cfg, tokenizer, validate=False)
    dataloaders = get_dataloaders(cfg, *datasets)

    model, optimizer = build_generator(cfg)
    trainer = create_trainer(
        cfg, model, optimizer, dataloaders[0].sampler, logging.getLogger()
    )

    print(dir(trainer))

"""Main methods for model training and evaluation."""

from importlib import import_module
from logging import Logger
from typing import Optional, Union

import ignite.distributed as idist
import torch
from catbird.core import Config  # type: ignore
from catbird.datasets import TeacherForcing
from ignite.contrib.engines import common
from ignite.engine import Engine
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


def default_train_step(
    cfg: Config,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: Union[str, torch.device],
    scaler: GradScaler,
):
    """Decorate train step to add more parameters.

    Args:
        cfg (Config): Config instance with configurations.
        model (nn.Module): a Pytorch model.
        optimizer (optim.Optimizer): a Pytorch optimizer.
        device (Union[str, torch.device]): specifies which device updates are accumulated on.
        scaler (GradScaler): GradScaler instance for gradient scaling.
    """

    def routine(engine, batch):
        accumulation_steps = cfg.train.get("accumulation_steps", 1)
        with_amp = cfg.train.get("with_amp", False)

        if cfg.data.get("tokenizer", None):
            ignore_index = -100
        else:
            ignore_index = cfg.pad_token_id
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)

        model.train()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True)
                for (k, v) in batch.items()
            }

        tgt = batch["tgt"]

        with autocast(enabled=with_amp):
            net_output, _ = model(**batch)

            loss = loss_fct(
                net_output.reshape(-1, net_output.size(-1)), tgt.reshape(-1)
            )

            loss /= accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    return routine



def default_evaluate_step(
    cfg: Config,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: Union[str, torch.device],
    logger: Logger,
):
    """Decorate evaluate step to add more parameters.

    Args:
        cfg (Config): Config instance with configurations.
        model (nn.Module): a Pytorch model.
        tokenizer (AutoTokenizer): a Pytorch optimizer.
        device (Union[str, torch.device]): specifies which device updates are accumulated on.
        logger (Logger): a Logger instance.
    """

    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return list(map(str.strip, gen_text))

    @torch.no_grad()
    def routine(engine, batch):
        model.eval()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        tgt = batch["tgt"]
        out = model.generate(src_ids)
        y_pred = torch.argmax(out, dim=-1)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(tgt)
        preds = [_preds.split() for _preds in preds]
        tgt = [[_tgt.split()] for _tgt in tgt]

        if engine.state.iteration % cfg.print_output_every == 0:
            logger.info(f'\n Preds : {" ".join(preds[0])} \n')
            logger.info(f'\n Target : {" ".join(tgt[0][0])} \n')
        return preds, tgt

    return routine


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

    if isinstance(cfg.model, dict) and "type" in cfg.model:
        train_step = default_train_step(cfg, model, optimizer, device, scaler)
    else:
        model_name = cfg.model.name.lower().split("-")[0]
        module = import_module(f"catbird.models.{model_name}")
        train_step = getattr(module, "train_step")(
            cfg, model, optimizer, device, scaler
        )

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

"""Main methods for model training and evaluation."""

from importlib import import_module
from logging import Logger
from typing import Optional, Union

import ignite.distributed as idist
import torch
from catbird.core import Config  # type: ignore
from ignite.contrib.engines import common
from ignite.engine import Engine
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import torch.nn.functional as F
from catbird.models.generators.base import Seq2Seq


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

        if cfg.data.get("mask_pad_token", False):
            ignore_index = -100
        else:
            ignore_index = cfg.pad_token_id

        model.train()

        if batch["tgt"].device != device:
            batch = {k: v.to(device) for (k, v) in batch.items()}

        with autocast(enabled=with_amp):
            loss = model(**batch, return_loss=True, ignore_index=ignore_index)

            loss /= accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    return routine


def default_evaluate_step(
    cfg: Config, model: nn.Module, tokenizer, logger,
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

        device = idist.device()
        if batch["tgt"].device != device:
            batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

        src_ids = batch["input_ids"]
        tgt = batch["tgt"]

        tgt = torch.where(tgt != -100, tgt, cfg.pad_token_id)

        if isinstance(model, Seq2Seq):
            y_pred = model.generate(src_ids, num_beams=1)
        else:
            y_pred = model.generate(src_ids)

        preds = ids_to_clean_text(y_pred)
        tgt_text = ids_to_clean_text(tgt)

        preds = [_preds.split() for _preds in preds]
        tgt_text = [[_tgt.split()] for _tgt in tgt_text]

        logger.info(
            f'\n Preds: {" ".join(preds[0])} \n\n Target: {" ".join(tgt_text[0][0])} \n'
        )

        logits = model(**batch, return_loss=False)

        output = logits.reshape(-1, logits.size(-1))
        target = tgt.reshape(-1)

        loss_fct = CrossEntropyLoss(ignore_index=cfg.pad_token_id)
        loss = loss_fct(output, target)
        nll = F.nll_loss(output, target, ignore_index=cfg.pad_token_id, reduction="sum")
        _, argmax = output.max(-1)
        invalid_targets = target.eq(cfg.pad_token_id)
        accuracy = argmax.eq(target).masked_fill_(
            invalid_targets, 0
        ).long().sum() / target.size(0)

        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "accuracy": accuracy.view(1, 1).item(),
        }

    return routine


def create_trainer(
    cfg: Config,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler,
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
        logger (Logger): object to log progress to the CLI.

    Returns:
        Engine: Object to run the defined train step over each batch of a dataset.
    """
    device = idist.device()
    scaler = GradScaler(enabled=cfg.train.with_amp)

    train_step = default_train_step(cfg, model, optimizer, device, scaler)

    trainer = Engine(train_step)
    trainer.logger = logger

    metric_names = ["batch loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        output_names=metric_names,
        lr_scheduler=lr_scheduler,
        clear_cuda_cache=False,
        with_pbars=True,
    )
    return trainer


def create_evaluator(
    cfg: Config, model: nn.Module, tokenizer: AutoTokenizer, logger: Logger,
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
    evaluate_step = default_evaluate_step(cfg, model, tokenizer, logger)

    evaluator = Engine(evaluate_step)

    return evaluator

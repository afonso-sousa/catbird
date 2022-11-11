"""Main methods for model training and evaluation."""

import random
from logging import Logger
from pathlib import Path
from typing import Dict, Optional, Union

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine
from nltk import word_tokenize
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from catbird.utils import Config, fopen


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
        model.train()

        accumulation_steps = cfg.train.get("accumulation_steps", 1)
        with_amp = cfg.train.get("with_amp", False)

        if batch["labels"].device != device:
            batch = {k: v.to(device) for (k, v) in batch.items()}

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        with autocast(enabled=with_amp):
            # if isinstance(model, EncoderDecoderBase):
            #     loss = model(input_ids=input_ids, labels=labels)[0]
            # else:
            #     loss = model(
            #         input_ids=input_ids,
            #         attention_mask=batch["attention_mask"],
            #         tgt=labels,
            #     )
            if cfg.data.get("with_dep", False):
                loss = model(input_ids=input_ids, graph=batch["graph"], labels=labels)[
                    0
                ]
                # loss = model(input_ids=input_ids, labels=labels)[0]
            else:
                loss = model(input_ids=input_ids, labels=labels)[0]

            loss /= accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    return routine


def default_evaluate_step(
    cfg: Config, model: nn.Module, tokenizer, logger, sample_save_path
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
        if batch["labels"].device != device:
            batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if cfg.data.get("with_dep", False):
            y_pred = model.generate(input_ids, batch["graph"])
        else:
            y_pred = model.generate(input_ids)

        src_ids_text = ids_to_clean_text(input_ids)
        preds_text = ids_to_clean_text(y_pred)
        tgt_text = ids_to_clean_text(labels)

        # Torch Ignite metrics expect a list of hypotheses and a list of lists of references
        src = [word_tokenize(_src) for _src in src_ids_text]
        preds = [word_tokenize(_preds) for _preds in preds_text]
        tgt = [[word_tokenize(_tgt)] for _tgt in tgt_text]

        if engine.state.iteration % cfg.test.print_output_every == 0:
            idx = random.randint(0, len(batch) - 1)
            output_info = (
                "\n"
                f"Random batch sample #{idx}\n"
                f"Source: {' '.join(src[idx])}\n"
                f"Preds: {' '.join(preds[idx])}\n"
                f"Target: {' '.join(tgt[idx][0])}\n"
            )

            logger.info(output_info)
            if sample_save_path:
                with fopen(
                    Path(sample_save_path)
                    / f"samples__{engine.state.trainer_epoch}_{engine.state.iteration}.txt",
                    mode="w",
                ) as f:
                    f.write(output_info)

        return {
            "y_pred": preds,
            "y": tgt,
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
    cfg: Config,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    metrics: Dict,
    logger: Logger,
    sample_save_path=None,
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
    evaluate_step = default_evaluate_step(
        cfg, model, tokenizer, logger, sample_save_path
    )

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

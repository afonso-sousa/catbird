"""Main methods for model training and evaluation."""

import random
from importlib import import_module
from logging import Logger
from typing import Optional, Union

import ignite.distributed as idist
import torch
import torch.nn.functional as F
from catbird.core import Config  # type: ignore
from catbird.models.generators.base import Seq2Seq
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
        model.train()

        # print(batch.keys())
        # print(batch)
        # assert False

        # accumulation_steps = cfg.train.get("accumulation_steps", 1)
        # with_amp = cfg.train.get("with_amp", False)

        if cfg.data.get("mask_pad_token", False):
            ignore_index = -100
        else:
            ignore_index = cfg.pad_token_id

        if batch["labels"].device != device:
            batch = {k: v.to(device) for (k, v) in batch.items()}

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # with autocast(enabled=with_amp):
        if isinstance(model, Seq2Seq):
            loss = model(input_ids=input_ids, labels=labels)[0]
        else:
            loss = model(input_ids=input_ids, attention_mask=batch["attention_mask"], tgt=labels)

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        # loss /= accumulation_steps

        # scaler.scale(loss).backward()

        # if engine.state.iteration % accumulation_steps == 0:
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()

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
        if batch["labels"].device != device:
            batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        labels = torch.where(labels != -100, labels, cfg.pad_token_id)

        idx = random.randint(0, len(batch) - 1)
        print("IDX " + str(idx))

        # if isinstance(model, Seq2Seq):
            # for i in range(len(batch)):
            #     y_pred = torch.stack(
            #         [
            #             model.generate(
            #                 input_ids[i].unsqueeze(0),
            #                 [list(cfg.bos_token_id)],
            #                 beam_size=3,
            #                 eos_id=cfg.eos_token_id,
            #             )
            #             for i in range(len(batch))
            #         ]
            #     )
            # y_pred = model.generate(src_ids, [[cfg.bos_token_id]] * len(batch), beam_size=1, eos_id=cfg.eos_token_id)
            # y_preds = model.generate(
            #     input_ids[0].unsqueeze(0),
            #     [[cfg.decoder_start_token_id]],
            #     beam_size=5,
            #     max_sequence_length=50,
            #     eos_id=cfg.eos_token_id,
            # )[0]
            # y_pred = [s.output for s in y_preds]
            
            # generated = model.generate(
            #     input_ids, max_length=50, num_beams=3, early_stopping=True
            # )
            # generated = model.generate(input_ids)

            # print(src_ids[0].unsqueeze(0).shape)
            # print(torch.tensor(y_pred[0]).unsqueeze(0).shape)
            # src_ids_text = ids_to_clean_text(input_ids[idx].unsqueeze(0))
            # preds = ids_to_clean_text(generated[idx].unsqueeze(0))
            # tgt_text = ids_to_clean_text(labels[idx].unsqueeze(0))
            # print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        # else:
        y_pred = model.generate(input_ids)

        src_ids_text = ids_to_clean_text(input_ids)
        preds = ids_to_clean_text(y_pred)
        tgt_text = ids_to_clean_text(labels)

        src_ids_text = [_src.split() for _src in src_ids_text]
        preds = [_preds.split() for _preds in preds]
        tgt_text = [_tgt.split() for _tgt in tgt_text]

        logger.info(
            "\n"
            f"Source: {' '.join(src_ids_text[0])}\n"
            f"Preds: {' '.join(preds[0])}\n"
            f"Target: {' '.join(tgt_text[0])}\n"
        )

        # loss = model(input_ids=input_ids, labels=labels,)[0]
        if isinstance(model, Seq2Seq):
            loss = model(input_ids=input_ids, labels=labels)[0]
        else:
            loss = model(input_ids=input_ids, attention_mask=batch["attention_mask"], tgt=labels)
        # logits = model(**batch, return_loss=False)

        # output = logits.reshape(-1, logits.size(-1))
        # target = labels[:, 1:].contiguous().reshape(-1)

        # loss_fct = CrossEntropyLoss(ignore_index=cfg.pad_token_id)
        # loss = loss_fct(output, target)
        # nll = F.nll_loss(output, target, ignore_index=cfg.pad_token_id, reduction="sum")
        # _, argmax = output.max(-1)
        # invalid_targets = target.eq(cfg.pad_token_id)
        # accuracy = argmax.eq(target).masked_fill_(
        #     invalid_targets, 0
        # ).long().sum() / target.size(0)

        return {
            "loss": loss.item(),
            # "nll": nll.item(),
            # "accuracy": accuracy.view(1, 1).item(),
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

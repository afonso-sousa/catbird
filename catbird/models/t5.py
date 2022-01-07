"""Method to initialize a pretrained T5 conditional generation model."""
from logging import Logger
from typing import Tuple, Union

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .utils import freeze_params


def train_step(
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
        
        model.train()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        src_attention_mask = batch["attention_mask"]
        tgt = batch["tgt"]

        with autocast(enabled=with_amp):
            y = model(input_ids=src_ids, attention_mask=src_attention_mask, labels=tgt)
            loss = y["loss"]
            loss /= accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    return routine


def evaluate_step(
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
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        src_attention_mask = batch["attention_mask"]
        tgt = batch["tgt"]
        if idist.get_world_size() > 1:
            y_pred = model.module.generate(
                input_ids=src_ids, attention_mask=src_attention_mask
            )
        else:
            y_pred = model.generate(
                input_ids=src_ids, attention_mask=src_attention_mask
            )

        tgt = torch.where(tgt != -100, tgt, tokenizer.pad_token_id)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(tgt)
        preds = [_preds.split() for _preds in preds]
        tgt = [[_tgt.split()] for _tgt in tgt]

        if engine.state.iteration % cfg.print_output_every == 0:
            logger.info(f'\n Preds : {" ".join(preds[0])} \n')
            logger.info(f'\n Target : {" ".join(tgt[0][0])} \n')
        return preds, tgt

    return routine


def initialize(cfg: Config) -> Tuple[nn.Module, optim.Optimizer]:
    """Initialize T5 conditional generator based on the given configurations.

    Args:
        cfg (Config): configuration file

    Returns:
        Tuple[nn.Module, optim.Optimizer]: model and optimizer
    """
    model = T5ForConditionalGeneration.from_pretrained(cfg.model.name)
    if cfg.get("tokenizer", None):
        model.resize_token_embeddings(cfg.embedding_length)

    lr = cfg.train.learning_rate * idist.get_world_size()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd_entry in n for nd_entry in no_decay)
            ],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd_entry in n for nd_entry in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if cfg.model.freeze_encoder:
        freeze_params(model.get_encoder())

    model = idist.auto_model(model)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = idist.auto_optim(optimizer)

    return model, optimizer

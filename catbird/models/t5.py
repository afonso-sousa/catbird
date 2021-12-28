"""Method to initialize a pretrained T5 conditional generation model."""

from typing import Tuple

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from catbird.core import Config  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from transformers import T5ForConditionalGeneration

from .utils import freeze_params


def train_step(cfg, model, optimizer, device, scaler):
    def routine(engine, batch):
        model.train()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        src_attention_mask = batch["attention_mask"]
        tgt = batch["tgt"]

        with autocast(enabled=cfg.train.with_amp):
            y = model(input_ids=src_ids, attention_mask=src_attention_mask, labels=tgt)
            loss = y["loss"]
            loss /= cfg.train.accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % cfg.train.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}
    
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

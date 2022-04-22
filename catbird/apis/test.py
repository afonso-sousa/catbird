"""Main methods for model evaluation and testing."""

from importlib import import_module
from logging import Logger
from typing import Union, Dict

import ignite.distributed as idist
import torch
from catbird.core import Config  # type: ignore
from ignite.engine import Engine
from torch import nn
from transformers import AutoTokenizer

from catbird.models.generators.base import Seq2Seq


def default_test_step(
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

        if batch["labels"].device != device:
            batch = {
                k: v.to(device, non_blocking=True)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        labels = batch["labels"]

        if isinstance(model, Seq2Seq):
            y_pred = model.generate(src_ids, num_beams = cfg.test.get("num_beams", 1))
        else:
            y_pred = model.generate(src_ids)

        if cfg.data.get("mask_pad_token", False):
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(labels)
        
        preds = [_preds.split() for _preds in preds]
        tgt = [[_tgt.split()] for _tgt in tgt]

        if engine.state.iteration % cfg.test.print_output_every == 0:
            logger.info(f'\n Preds : {" ".join(preds[0])} \n')
            logger.info(f'\n Target : {" ".join(tgt[0][0])} \n')
        return preds, tgt

    return routine


def create_tester(
    cfg: Config,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    metrics: Dict,
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

    test_step = default_test_step(cfg, model, tokenizer, device, logger)

    tester = Engine(test_step)
    
    for name, metric in metrics.items():
        metric.attach(tester, name)

    return tester
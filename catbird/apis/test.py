from logging import Logger
from typing import Dict, Union

import ignite.distributed as idist
import torch
from catbird.utils import Config
from ignite.engine import Engine
from nltk import word_tokenize
from torch import nn
from transformers import AutoTokenizer


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
            batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # if isinstance(model, EncoderDecoderBase):
        #     y_pred = model.generate(src_ids, num_beams=cfg.test.get("num_beams", 1))
        # else:
        # y_pred = model.generate(src_ids)
        if cfg.data.get("with_dep", False):
            y_pred = model.generate(input_ids, batch["graph"])
        else:
            y_pred = model.generate(input_ids)

        preds_text = ids_to_clean_text(y_pred)
        tgt_text = ids_to_clean_text(labels)

        preds = [word_tokenize(_preds) for _preds in preds_text]
        tgt = [[word_tokenize(_tgt)] for _tgt in tgt_text]

        if engine.state.iteration % cfg.test.print_output_every == 0:
            logger.info(
                (
                    "\n"
                    f'\n Preds: {" ".join(preds[0])}\n'
                    f'\n Target: {" ".join(tgt[0][0])}\n'
                )
            )
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

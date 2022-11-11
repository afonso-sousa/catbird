"""File to test a paraphrase generator."""
import argparse
from pathlib import Path

import torch
from catbird.apis import create_evaluator

# from catbird.core import TER, Meteor
from catbird.utils import Config, mkdir_or_exist, log_metrics
from catbird.datasets import build_dataset, get_dataloader
from catbird.models import build_generator_model
from catbird.tokenizers import build_tokenizer
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.metrics import Bleu, RougeN
from ignite.utils import setup_logger

available_metrics = {
    "bleu-4": Bleu(ngram=4),
    # "bleu_smooth_2": Bleu(ngram=4, smooth="smooth2", average="micro"),
    # "meteor": Meteor(),
    # "ter": TER(),
    "rouge-2": RougeN(ngram=2)["Rouge-2-F"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test (and eval) a paraphrase generator"
    )
    parser.add_argument("config", help="train config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--metrics",
        type=str,
        default=["bleu-4", "rouge-2"],
        choices=["bleu-4", "rouge-2", "meteor", "ter"],
        nargs="+",
        help="evaluation metrics",
    )
    parser.add_argument("--work-dir", help="dir to save logs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = Path("./work_dirs", Path(args.config).stem)

    mkdir_or_exist(Path(cfg.work_dir).resolve())

    logger = setup_logger(name="Test")

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)

    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.sep_token_id
    cfg.bos_token_id = tokenizer.cls_token_id
    cfg.decoder_start_token_id = cfg.bos_token_id
    logger.info(
        f"Tokenizer special tokens: \npad_token: {tokenizer.decode(cfg.pad_token_id)} - {cfg.pad_token_id}\n"
        f"eos_token: {tokenizer.decode(cfg.eos_token_id)} - {cfg.eos_token_id}\n"
        f"decoder_start_token: {tokenizer.decode(cfg.decoder_start_token_id)} - {cfg.decoder_start_token_id}"
    )

    val_dataset = build_dataset(cfg, "val", tokenizer)
    val_dataloader = get_dataloader(cfg, "test", val_dataset)

    # cfg.resume_from = args.checkpoint

    model = build_generator_model(cfg)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.info(f"Model's # of parameters: {num_parameters}")

    checkpoint = torch.load(args.checkpoint)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    selected_metrics = {
        key: value for key, value in available_metrics.items() if key in args.metrics
    }
    logger.info(f"The selected metrics are: {', '.join(selected_metrics.keys())}")

    tester = create_evaluator(cfg, model, tokenizer, selected_metrics, logger)

    # Compute intermediate metrics
    @tester.on(Events.ITERATION_COMPLETED(every=cfg.test.print_output_every))
    def compute_and_measure():
        [value.completed(tester, key) for key, value in selected_metrics.items()]

    @tester.on(
        Events.ITERATION_COMPLETED(every=cfg.test.print_output_every) | Events.TERMINATE
    )
    def log_info():
        log_metrics(
            logger, tester.state.times["COMPLETED"], "Testing", tester.state.metrics
        )

    common.ProgressBar(persist=False).attach(tester)

    try:
        state = tester.run(
            val_dataloader,
        )
    except Exception as e:
        logger.exception("")
        raise e


if __name__ == "__main__":
    main()

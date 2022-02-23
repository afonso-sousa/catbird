"""File to test a paraphrase generator."""
import argparse
from pathlib import Path

import torch
from catbird.apis import create_evaluator
from catbird.core import TER, Config, Meteor, mkdir_or_exist
from catbird.datasets import build_dataset, get_dataloader
from catbird.models import build_generator
from catbird.tokenizers import build_tokenizer
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.metrics import Bleu
from ignite.utils import setup_logger

available_metrics = {
    "bleu": Bleu(ngram=4, smooth="smooth1", average="micro"),
    "bleu_smooth_2": Bleu(ngram=4, smooth="smooth2", average="micro"),
    "meteor": Meteor(),
    "ter": TER(),
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
        default=["bleu", "bleu_smooth_2"],
        choices=["bleu", "bleu_smooth_2", "meteor", "ter"],
        nargs="+",
        help="evaluation metrics",
    )
    parser.add_argument("--work-dir", help="dir to save logs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # group_gpus = parser.add_mutually_exclusive_group()
    # group_gpus.add_argument("--gpus", type=int, help="number of gpus to use")
    # group_gpus.add_argument("--gpu-ids", type=int, nargs="+", help="ids of gpus to use")

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

    val_dataset = build_dataset(cfg, "val", tokenizer)
    val_dataloader = get_dataloader(cfg, "val", val_dataset)

    model = build_generator(cfg)
    
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.info(f"Model's # of parameters: {num_parameters}")

    checkpoint = torch.load(args.checkpoint)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    selected_metrics = {
        key: value for key, value in available_metrics.items() if key in args.metrics
    }
    evaluator = create_evaluator(cfg, model, tokenizer, selected_metrics, logger)

    @evaluator.on(Events.COMPLETED)
    def log_info():
        metrics_output = "\n".join(
            [f"\t{k}: {v}" for k, v in evaluator.state.metrics.items()]
        )

        logger.info(
            f"\nEvaluation time (seconds): {evaluator.state.times['COMPLETED']:.2f}\nValidation metrics:\n {metrics_output}"
        )

    common.ProgressBar(persist=False).attach(evaluator)

    try:
        state = evaluator.run(val_dataloader,)
    except Exception as e:
        logger.exception("")
        raise e


if __name__ == "__main__":
    main()

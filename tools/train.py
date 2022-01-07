"""File to train a paraphrase generator."""
import argparse
import warnings
from pathlib import Path

import ignite.distributed as idist
from catbird.apis import create_evaluator, create_trainer
from catbird.core import (Config, log_basic_info, log_metrics_eval,
                          mkdir_or_exist)
from catbird.datasets import build_dataset, get_dataloader
from catbird.models import build_generator
from catbird.tokenizers import build_tokenizer
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Bleu
from ignite.utils import manual_seed, setup_logger

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a paraphrase generator")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--gpus", type=int, help="number of gpus to use")
    group_gpus.add_argument("--gpu-ids", type=int, nargs="+", help="ids of gpus to use")

    args = parser.parse_args()

    return args


def training(local_rank, cfg, args):
    rank = idist.get_rank()
    manual_seed(args.seed + rank)
    device = idist.device()

    logger = setup_logger(name="Train", distributed_rank=local_rank)
    log_basic_info(logger, cfg)

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id

    train_dataset = build_dataset(cfg, "train", tokenizer)
    train_dataloader = get_dataloader(cfg, "train", train_dataset)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        logger.info(f"Resuming model from '{cfg.resume_from}'")
    model, optimizer = build_generator(cfg)

    trainer = create_trainer(cfg, model, optimizer, train_dataloader.sampler, logger)

    metrics = {
        "bleu": Bleu(ngram=4, smooth="smooth1", average="micro"),
        "bleu_smooth_2": Bleu(ngram=4, smooth="smooth2", average="micro"),
    }

    best_model_handler = Checkpoint(
        {"model": model},
        DiskSaver(dirname=cfg.work_dir.as_posix(), require_empty=False),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_bleu",
        score_function=Checkpoint.get_default_score_fn("bleu"),
    )

    if not args.no_validate:
        val_dataset = build_dataset(cfg, "val", tokenizer)
        val_dataloader = get_dataloader(cfg, "val", val_dataset)

        evaluator = create_evaluator(cfg, model, tokenizer, metrics, logger)
        evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

        @trainer.on(Events.EPOCH_COMPLETED(every=1) | Events.COMPLETED | Events.STARTED)
        def run_validation():
            epoch = trainer.state.epoch
            state = evaluator.run(val_dataloader)
            log_metrics_eval(
                logger, epoch, state.times["COMPLETED"], "Validation", state.metrics
            )

    if rank == 0:
        evaluators = {"val": evaluator} if (not args.no_validate) else None
        tb_logger = common.setup_tb_logging(
            cfg.work_dir, trainer, optimizer, evaluators=evaluators
        )

    try:
        state = trainer.run(
            train_dataloader,
            max_epochs=cfg.train.num_epochs,
            epoch_length=cfg.train.epoch_length,
        )
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def run():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = Path("./work_dirs", Path(args.config).stem)

    mkdir_or_exist(Path(cfg.work_dir).resolve())

    with idist.Parallel(backend=None, nproc_per_node=None) as parallel:
        parallel.run(training, cfg, args)


if __name__ == "__main__":
    run()

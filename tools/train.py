import argparse
import warnings
from pathlib import Path

import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Bleu
from ignite.utils import manual_seed, setup_logger
from catbird.apis import create_evaluator, create_trainer
from catbird.core import (Config, log_basic_info, log_metrics_eval,
                           mkdir_or_exist)
from catbird.datasets import build_dataset, get_dataloaders
from catbird.models import build_generator
from transformers import AutoTokenizer

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

    logger = setup_logger(name="Paraphrase", distributed_rank=local_rank)
    log_basic_info(logger, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    datasets = build_dataset(cfg, tokenizer, validate=(not args.no_validate))
    dataloaders = get_dataloaders(cfg, *datasets)

    model, optimizer = build_generator(cfg)

    trainer = create_trainer(cfg, model, optimizer, dataloaders[0].sampler, logger)

    metrics = {
        "bleu": Bleu(ngram=4, smooth="smooth1", average="micro"),
        "bleu_smooth_2": Bleu(ngram=4, smooth="smooth2", average="micro"),
    }

    if not args.no_validate:
        evaluator = create_evaluator(cfg, model, tokenizer, metrics, logger)

        @trainer.on(Events.EPOCH_COMPLETED(every=1) | Events.COMPLETED | Events.STARTED)
        def run_validation(engine):
            epoch = trainer.state.epoch
            state = evaluator.run(dataloaders[1])
            log_metrics_eval(
                logger, epoch, state.times["COMPLETED"], "Validation", state.metrics
            )

    if rank == 0:
        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # folder_name = f"paraphrase_model_backend-{idist.backend()}-{idist.get_world_size()}_{timestamp}"
        # output_path = Path(cfg.work_dir, folder_name)
        # mkdir_or_exist(output_path)
        # logger.info(f"Output path: {output_path}")

        evaluators = {"val": evaluator} if (not args.no_validate) else None
        tb_logger = common.setup_tb_logging(
            cfg.work_dir, trainer, optimizer, evaluators=evaluators
        )

    best_model_handler = Checkpoint(
        {"model": model},
        cfg.work_dir.as_posix(),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_bleu",
        score_function=Checkpoint.get_default_score_fn("bleu"),
    )
    if not args.no_validate:
        evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    try:
        state = trainer.run(
            dataloaders[0],
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

    import os.path as osp

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = Path("./work_dirs", Path(args.config).stem)

    mkdir_or_exist(Path(cfg.work_dir).resolve())

    with idist.Parallel(backend=None, nproc_per_node=None) as parallel:
        parallel.run(training, cfg, args)


if __name__ == "__main__":
    run()

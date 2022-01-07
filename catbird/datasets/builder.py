"""Factory to build datasets."""

from pathlib import Path
from typing import Tuple

import ignite.distributed as idist
from catbird.core import Config, load  # type: ignore
from ignite.utils import setup_logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .quora import QuoraDataset as QuoraDataset


def build_dataset(cfg: Config, split: str, tokenizer: AutoTokenizer) -> Tuple[Dataset]:
    """Abstraction to build datasets based on the given configurations.

    Args:
        cfg (Config): configuration file
        split (str): split name. Either train or val
        tokenizer (AutoTokenizer): HuggingFace tokenizer

    Raises:
        NameError: Exception raised if dataset name in config does not match any of the available options

    Returns:
        tuple: tuple with train dataset or train and validation datasets
    """
    logger = setup_logger(name="Dataset", distributed_rank=idist.get_rank())
    logger.info(f"Loading {cfg.dataset_name} dataset")
    [logger.info(f"{k} - {v}") for k, v in cfg.data.items()]

    if cfg.dataset_name.lower() == "quora":
        data = load(Path(cfg.data_root) / f"quora_{split}.pkl")
        dataset = QuoraDataset(cfg, split, data, tokenizer)
        return dataset
    else:
        raise NameError(
            "The dataset name does not match any of our currently available options."
        )


def get_dataloader(cfg: Config, split: str, dataset: Dataset) -> DataLoader:
    """Get dataloaders of given datasets.

    Args:
        cfg (Config): configuration file
        split (str): split name. Either train or val
        dataset (Dataset): Pytorch Dataset instance

    Returns:
        DataLoader: tuple with train or train and validation dataloaders
    """
    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    loader = idist.auto_dataloader(
        dataset,
        batch_size=cfg.train.batch_size * (1 if split == "train" else 2),
        num_workers=cfg.num_workers,
        shuffle=(True if split == "train" else False),
        drop_last=(True if split == "train" else False),
    )
    return loader

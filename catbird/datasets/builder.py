"""Factory to build datasets."""

from typing import Any, Tuple

import ignite.distributed as idist
from catbird.core import Config, load  # type: ignore
from ignite.utils import setup_logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from pathlib import Path

from .quora import QuoraDataset as QuoraDataset


def build_dataset(
    cfg: Config, tokenizer: AutoTokenizer, **kwargs: Any
) -> Tuple[Dataset]:
    """Abstraction to build datasets based on the given configurations.

    Args:
        cfg (Config): configuration file
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
        train_data = load(Path(cfg.data_root) / "quora_train.pkl")
        train_dataset = QuoraDataset(cfg, "train", train_data, tokenizer)
        if kwargs.pop("validate", False):
            val_data = load(Path(cfg.data_root) / "quora_val.pkl")
            val_dataset = QuoraDataset(cfg, "val", val_data, tokenizer)
            return train_dataset, val_dataset
        else:
            return (train_dataset,)
    else:
        raise NameError(
            "The dataset name does not match any of our currently available options."
        )


def get_dataloaders(
    cfg: Config, train_dataset: Dataset, val_dataset: Dataset = None
) -> Tuple[DataLoader]:
    """Get dataloaders of given datasets.

    Args:
        cfg (Config): configuration file
        train_dataset (Dataset): train dataset
        val_dataset (Dataset, optional): validation dataset. Defaults to None.

    Returns:
        Tuple[DataLoader]: tuple with train or train and validation dataloaders
    """
    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if val_dataset:
        val_loader = idist.auto_dataloader(
            val_dataset,
            batch_size=2 * cfg.train.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        return (train_loader, val_loader)  # type: ignore
    else:
        return (train_loader,)  # type: ignore

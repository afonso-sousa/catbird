"""Factory to build datasets."""

from typing import Any, Tuple

import ignite.distributed as idist
from catbird.core import Config  # type: ignore
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datasets import load_dataset  # type: ignore

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
    if cfg.dataset_name.lower() == "quora":
        dataset = load_dataset("quora")
        dataset = dataset.shuffle(seed=kwargs.pop("seed", 0))
        dataset = dataset["train"]
        dataset = dataset.filter(
            lambda example: example["is_duplicate"] == True
        )  # Filter for true paraphrases
        if kwargs.pop("validate", False):
            dataset = dataset.train_test_split(test_size=cfg.data.val.train_test_split)
            return (
                QuoraDataset(cfg, "train", dataset["train"], tokenizer),
                QuoraDataset(cfg, "val", dataset["test"], tokenizer),
            )
        else:
            return (QuoraDataset(cfg, "train", dataset, tokenizer),)
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

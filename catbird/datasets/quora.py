"""Define Torch Dataset for Quora Questions Pairs corpus."""

from typing import Dict

import torch
from catbird.core import Config  # type: ignore
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets import Dataset as HugDataset


class QuoraDataset(Dataset):
    """Torch Dataset class to process the Quora dataset."""

    def __init__(
        self, cfg: Config, split: str, data: HugDataset, tokenizer: AutoTokenizer
    ) -> None:
        """Initialize attributes of the class.

        Args:
            cfg (Config): configuration file
            split (str): string with either 'train' or 'val' to guide configuration attributes to use
            data (HugDataset): Dataset instance from HuggingFace
            tokenizer (AutoTokenizer): AutoTokenizer instance from HuggingFace
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = cfg.data.max_length
        self.length_dataset = (
            cfg.data[split].dataset_length
            if cfg.data[split].dataset_length != -1
            else len(self.data)
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset record in index.

        Args:
            idx (int): index of entry to retrieve

        Returns:
            dict[str, torch.Tensor]: dictionary with src and target sentences as embedding tensors
        """
        # t5 models require a prefix describing the task
        task_prefix = "paraphrase: "
        src_text = [task_prefix + str(self.data[idx]["questions"]["text"][0]) + " </s>"]

        tgt_text = [str(self.data[idx]["questions"]["text"][1])]
        input_txt_tokenized = self.tokenizer(
            src_text, max_length=self.max_length, padding="max_length", truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            tgt_text_tokenized = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        # The pad token in target is replaced with -100 so that it doesn't get added to loss.
        tgt_text_tokenized = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in tgt_text_tokenized.input_ids
        ]

        input_txt_tokenized.update({"tgt": tgt_text_tokenized[0]})

        batch = {
            k: torch.tensor(v).squeeze(0) for (k, v) in input_txt_tokenized.items()
        }
        return batch

    def __len__(self) -> int:
        """Give total number of records in dataset.

        Returns:
            int: length of dataset.
        """
        return self.length_dataset

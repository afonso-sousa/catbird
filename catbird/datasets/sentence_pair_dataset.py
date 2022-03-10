"""Define Torch Dataset for Quora Questions Pairs corpus."""

from typing import Dict, List

import torch
from catbird.core import Config
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SentencePairDataset(Dataset):
    """Torch Dataset class to process sentence pair datasets."""

    def __init__(
        self, cfg: Config, split: str, data: List, tokenizer: AutoTokenizer
    ) -> None:
        """Initialize attributes of the class.

        Args:
            cfg (Config): configuration file
            split (str): string with either 'train' or 'val' to guide configuration attributes to use
            data (List): List of dictionaries with reference and candidate sentences
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
        self.mask_pad_token = cfg.data.get("mask_pad_token", False)
        self.task_prefix = cfg.data.get("task_prefix", "")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset record in index.

        Args:
            idx (int): index of entry to retrieve

        Returns:
            dict[str, torch.Tensor]: dictionary with src and target sentences as embedding tensors
        """

        # input ids
        src_text = [self.task_prefix + str(self.data[idx]["src"])]

        input_txt_tokenized = self.tokenizer(
            src_text, max_length=self.max_length, padding="max_length", truncation=True
        )

        # tgt
        tgt_text = [str(self.data[idx]["tgt"])]
        with self.tokenizer.as_target_tokenizer():
            tgt_text_tokenized = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        # If mask_pad_token, the pad token in target is replaced with -100 so that it doesn't get added to loss.
        if self.mask_pad_token:
            tgt_text_tokenized = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in tgt_text_tokenized.input_ids
            ]
        else:
            tgt_text_tokenized = tgt_text_tokenized.input_ids

        input_txt_tokenized.update({"tgt": tgt_text_tokenized[0]})

        batch = {
            k: torch.tensor(v).squeeze(0) for (k, v) in input_txt_tokenized.items()
        }
        
        batch["src_lengths"] = torch.sum(
            batch["input_ids"].ne(self.tokenizer.pad_token_id)
        )

        return batch

    def __len__(self) -> int:
        """Give total number of records in dataset.

        Returns:
            int: length of dataset.
        """
        return self.length_dataset

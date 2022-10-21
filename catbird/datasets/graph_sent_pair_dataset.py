"""Define Torch Dataset for Quora Questions Pairs corpus."""

from functools import partial
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from catbird.utils import Config

from .sentence_pair_dataset import SentencePairDataset
from .graph_utils import build_levi_graph


class GraphSentPairDataset(SentencePairDataset):
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
        super(GraphSentPairDataset, self).__init__(cfg, split, data, tokenizer)
        self.all_dependencies = list(
            set(token["dep"] for sample in data for token in sample["src_dp"])
        )
        self.all_pos = list(
            set(token["pos"] for sample in data for token in sample["src_dp"])
        )
        self.num_relations = len(self.all_dependencies)

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset record in index.

        Args:
            idx (int): index of entry to retrieve

        Returns:
            dict[str, torch.Tensor]: dictionary with src and target sentences as embedding tensors
        """
        batch = super().__getitem__(idx)

        # tokenizer = partial(
        #     self.tokenizer, max_length=16, padding="max_length", truncation=True
        # )
        # batch["graph"] = build_dependency_structure(
        #     self.data[idx], self.all_dependencies, self.all_pos, self.tokenizer
        # )

        batch["graph"] = build_levi_graph(
            self.data[idx], self.all_dependencies, self.all_pos, self.tokenizer
        )

        return batch

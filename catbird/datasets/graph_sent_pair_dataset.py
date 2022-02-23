"""Define Torch Dataset for Quora Questions Pairs corpus."""

from typing import Dict, List

import torch
from catbird.core import Config  # type: ignore
from transformers import AutoTokenizer
from functools import partial

from .utils import build_levi_graph
from .sentence_pair_dataset import SentencePairDataset


class GraphSentPairDataset(SentencePairDataset):
    """Torch Dataset class to process sentence pair datasets."""

    def __init__(
        self, cfg: Config, split: str, data: List, tokenizer: AutoTokenizer, graph: List
    ) -> None:
        """Initialize attributes of the class.

        Args:
            cfg (Config): configuration file
            split (str): string with either 'train' or 'val' to guide configuration attributes to use
            data (List): List of dictionaries with reference and candidate sentences
            tokenizer (AutoTokenizer): AutoTokenizer instance from HuggingFace
        """
        super(GraphSentPairDataset, self).__init__(cfg, split, data, tokenizer)
        self.graph = graph

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset record in index.

        Args:
            idx (int): index of entry to retrieve

        Returns:
            dict[str, torch.Tensor]: dictionary with src and target sentences as embedding tensors
        """
        batch = super().__getitem__(idx)
        
        tokenizer = partial(self.tokenizer, max_length=16, padding="max_length", truncation=True)
        batch["ie_graph"] = build_levi_graph(self.graph[idx], tokenizer)

        return batch

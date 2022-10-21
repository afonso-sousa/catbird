from pathlib import Path
import unittest
from catbird.utils import Config
import torch

from catbird.datasets import build_dataset, get_dataloader
from catbird.tokenizers import build_tokenizer

data_path = Path(__file__).parent.parent.parent / "data"


def ids_to_clean_text(generated_ids, tokenizer):
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return list(map(str.strip, gen_text))


class TestBaseGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg_dict = dict(
            num_workers=4,
            dataset_name="Quora",
            data_root=data_path / "quora",
            data=dict(
                max_length=40,
                train=dict(dataset_length=-1),
                val=dict(dataset_length=-1),
                with_dep=True,
            ),
            tokenizer=dict(name="roberta-base"),
            train=dict(
                num_epochs=100,
                batch_size=4,
            ),
        )
        cls.cfg = Config(cfg_dict)

        cls.tokenizer = build_tokenizer(cls.cfg)

        cls.graph_valset = build_dataset(cls.cfg, "val", cls.tokenizer)
        cls.graph_valloader = get_dataloader(cls.cfg, "val", cls.graph_valset)
        cls.cfg.data.with_dep = False
        cls.regular_valset = build_dataset(cls.cfg, "val", cls.tokenizer)
        cls.regular_valloader = get_dataloader(cls.cfg, "val", cls.regular_valset)

    def test_quora_graph_dataset(self):
        graph_valset_sample = self.graph_valset[2]
        regular_valset_sample = self.regular_valset[2]

        assert torch.all(
            regular_valset_sample["input_ids"].eq(graph_valset_sample["input_ids"])
        )

    def test_quora_graph_dataloader(self):
        graph_sample = next(iter(self.graph_valloader))
        regular_sample = next(iter(self.regular_valloader))

        assert torch.all(
            graph_sample["input_ids"].eq(regular_sample["input_ids"])
        )  # validation dataloader is not shuffled

        assert torch.all(graph_sample["labels"].eq(regular_sample["labels"]))

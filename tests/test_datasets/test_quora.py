import unittest

from catbird.core import Config
from catbird.datasets import build_dataset, get_dataloader
from catbird.tokenizers import build_tokenizer


class TestEDD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.cfg = Config.fromfile("configs/edd_quora.yaml")
        cfg_dict = dict(
            num_workers=4,
            dataset_name="Quora",
            data_root="data/quora/",
            data=dict(
                max_length=80,
                train=dict(dataset_length=-1),
                val=dict(dataset_length=2000),
                task_prefix="paraphrase: ",
            ),
            train=dict(batch_size=32),
            model=dict(name="t5-small"),
        )
        cls.cfg = Config(cfg_dict)

        cls.tokenizer = build_tokenizer(cls.cfg)
        cls.cfg.embedding_length = len(cls.tokenizer)
        cls.cfg.pad_token_id = cls.tokenizer.pad_token_id

    def test_quora_train_batches(self):
        train_dataset = build_dataset(self.cfg, "train", self.tokenizer)
        train_len = self.cfg.data.train.dataset_length
        assert len(train_dataset) == train_len if train_len != -1 else 104484
        train_loader = get_dataloader(self.cfg, "train", train_dataset)

        sample_batch = next(iter(train_loader))
        src_ids = sample_batch["input_ids"]
        src_tokens = self.tokenizer.convert_ids_to_tokens(src_ids[0])
        assert src_tokens[:3] == ["▁para", "phrase", ":"]

        for sample_batch in train_loader:
            src_ids = sample_batch["input_ids"]
            tgt = sample_batch["tgt"]

            assert src_ids.shape == (
                self.cfg.train.batch_size,
                self.cfg.data.max_length,
            )
            assert src_ids.shape == tgt.shape

        # print(src_tokens)
        # assert False

    def test_quora_val_batches(self):
        val_dataset = build_dataset(self.cfg, "val", self.tokenizer)
        val_len = self.cfg.data.val.dataset_length
        assert len(val_dataset) == val_len if val_len != -1 else 44779
        val_loader = get_dataloader(self.cfg, "val", val_dataset)
        num_val_batches = len(list(val_loader))
        for i, sample_batch in enumerate(val_loader):
            src_ids = sample_batch["input_ids"]
            tgt = sample_batch["tgt"]

            if i + 1 != num_val_batches:
                assert src_ids.shape == (
                    2 * self.cfg.train.batch_size,
                    self.cfg.data.max_length,
                )
                assert src_ids.shape == tgt.shape


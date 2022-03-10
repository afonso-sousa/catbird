import unittest

import torch
from catbird.core import Config
from catbird.datasets import build_dataset, get_dataloader
from catbird.tokenizers import build_tokenizer
import copy


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg_dict = dict(
            num_workers=4,
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
        self.cfg.dataset_name = "Quora"
        self.cfg.data_root = "data/quora/"
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

        prev_output_tokens = sample_batch["prev_output_tokens"]

        if self.tokenizer.eos_token_id is None:
            assert torch.all(
                prev_output_tokens.eq(
                    torch.cat((src_ids[:, -1:], src_ids[:, :-1]), dim=1)
                )
            )
        else:
            assert torch.all(
                prev_output_tokens.eq(
                    torch.cat((torch.full((src_ids.size(0), 1), self.tokenizer.eos_token_id), src_ids[:, :-1]), dim=1)
                )
            )
            
    def test_prev_output_tokens(self):
        cfg_dict = dict(
            num_workers=4,
            data=dict(
                max_length=80,
                train=dict(dataset_length=-1),
                val=dict(dataset_length=2000),
            ),
            train=dict(batch_size=32),
            model=dict(name="t5-small"),
        )
        cfg = Config(cfg_dict)
        cfg.dataset_name = "Quora"
        cfg.data_root = "data/quora/"
        tokenizer = build_tokenizer(cfg)
        val_dataset = build_dataset(cfg, "val", tokenizer)
        val_loader = get_dataloader(cfg, "val", val_dataset)
        sample_batch = next(iter(val_loader))
        input_ids = sample_batch["input_ids"]
        prev_output_tokens = sample_batch["prev_output_tokens"]
        tgt = sample_batch["tgt"]
        
        t = tokenizer.decode(prev_output_tokens[0], skip_special_tokens=False)
        print(t)
        print(input_ids)
        print(_shift_right(input_ids, tokenizer.eos_token_id))
        assert False
        
   
        if tokenizer.eos_token_id is None:
            assert torch.all(
                prev_output_tokens.eq(
                    torch.cat((input_ids[:, -1:], tgt[:, :-1]), dim=1)
                )
            )
        else:
            assert torch.all(
                prev_output_tokens.eq(
                    torch.cat((torch.full((input_ids.size(0), 1), tokenizer.eos_token_id), tgt[:, :-1]), dim=1)
                )
            )





    def test_quora_val_batches(self):
        self.cfg.dataset_name = "Quora"
        self.cfg.data_root = "data/quora/"
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

    def test_mscoco_train_batches(self):
        self.cfg.dataset_name = "MSCOCO"
        self.cfg.data_root = "data/mscoco/"
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

    def test_mscoco_val_batches(self):
        self.cfg.dataset_name = "MSCOCO"
        self.cfg.data_root = "data/mscoco/"
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

    def test_graphs_in_batches(self):
        self.cfg.dataset_name = "MSCOCO"
        self.cfg.data_root = "data/mscoco/"
        self.cfg.data.use_ie_graph = True
        val_dataset = build_dataset(self.cfg, "val", self.tokenizer)
        
        val_loader = get_dataloader(self.cfg, "val", val_dataset)
        sample_batch = next(iter(val_loader))

        print(sample_batch["ie_graph"])
        assert False
    
    def test_graph_collate(self):
        from torch.utils.data.dataloader import default_collate
        from torch_geometric.data import Batch, Data
        from torch_geometric.loader import DataLoader as PyGDataLoader

        batch = default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        # {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        print(batch)
        
        edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index)
        s1 = {'input_ids': torch.randint(1, 10, (5,)),
                  'attention_mask': torch.zeros(5),
                  'tgt': torch.randint(1, 10, (5,)),
                  'prev_output_tokens': torch.randint(1, 10, (5,)),
                  'src_lengths': torch.tensor(5),
                  'ie_graph': graph}
        
        s2 = copy.deepcopy(s1)
        sample = [s1, s2]

        # batch2 = [default_collate(s) for s in sample if not isinstance(s, Data)]
        # print(batch2)
        loader = PyGDataLoader(sample, batch_size=32, shuffle=True)
        print(next(iter(loader)))
        assert False

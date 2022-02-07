import unittest

import torch
from catbird.core import Config
from catbird.datasets import build_dataset, get_dataloader
from catbird.models.edd import EDD
from catbird.models.losses import sent_emb_loss
from catbird.tokenizers import build_tokenizer


class TestEDD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile("configs/edd_quora.yaml")

        cls.tokenizer = build_tokenizer(cls.cfg)
        cls.cfg.embedding_length = len(cls.tokenizer)
        cls.cfg.pad_token_id = cls.tokenizer.pad_token_id

        dataset = build_dataset(cls.cfg, "val", cls.tokenizer)
        dataloader = get_dataloader(cls.cfg, "val", dataset)
        sample_batch = next(iter(dataloader))  # [train.batch_size, data.max_length]
        cls.src_ids = sample_batch["input_ids"]
        cls.tgt = sample_batch["tgt"]

        cls.model = EDD(cls.cfg)

    def test_edd_train(self):
        out, enc_out, enc_sim_phrase = self.model(self.src_ids, self.tgt)

        assert out.shape == (
            self.cfg.train.batch_size * 2,
            self.cfg.data.max_length,
            self.cfg.embedding_length,
        )

        if self.cfg.data.get("mask_pad_token", None):
            ignore_index = -100
        else:
            ignore_index = self.cfg.pad_token_id
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        out, enc_out, enc_sim_phrase = self.model(self.src_ids, self.tgt)
        print(out.shape)

        _ = loss_fct(out.reshape(-1, out.size(-1)), self.tgt.reshape(-1))
        _ = sent_emb_loss(enc_out, enc_sim_phrase)

    def test_edd_val(self):
        def ids_to_clean_text(generated_ids):
            gen_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return list(map(str.strip, gen_text))

        out = self.model.generate(self.src_ids)
        y_pred = torch.argmax(out, dim=-1)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(self.tgt)
        preds = [_preds.split() for _preds in preds]
        tgts = [[_tgt.split()] for _tgt in tgt]

        assert len(preds) == len(tgts)

    def test_edd_structure(self):
        encoder = self.model.encoder
        input = torch.randint(0, 10000, (32, 80))
        target = torch.randint(0, 10000, (32, 80))
        encoded_input = encoder(input)

        assert encoded_input.shape == (1, 32, 512)

        out, enc_out, enc_tgt = self.model(input, target)

        assert out.shape == (
            self.cfg.train.batch_size,
            self.cfg.data.max_length,
            self.cfg.embedding_length,
        )

        assert (
            enc_out.shape
            == enc_tgt.shape
            == (self.cfg.train.batch_size, self.cfg.model.emb_dim)
        )

        out = self.model.generate(self.src_ids)

        assert out.shape == (
            2 * self.cfg.train.batch_size,
            self.cfg.data.max_length,
            self.cfg.embedding_length,
        )

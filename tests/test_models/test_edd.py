import unittest

import torch
from catbird.core import Config
from catbird.datasets import build_dataset, get_dataloader
from catbird.models.edd import EDD
from catbird.models.losses import pair_wise_loss
from catbird.tokenizers import build_tokenizer
from catbird.core.utils.registry import build_from_cfg
from catbird.models.registry import GENERATORS


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
        _ = pair_wise_loss(enc_out, enc_sim_phrase)

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
    
    def test_edd_structure2(self):
        batch_size = 32
        max_length = 80
    
        input_ids = torch.randint(1, 100, (batch_size, max_length))
        src_lengths = torch.randint(5, 15, (batch_size,))

        mask = torch.zeros(input_ids.shape[0], input_ids.shape[1])
        mask[(torch.arange(input_ids.shape[0]), src_lengths)] = 1
        mask = mask.cumsum(dim=1)

        input_ids = input_ids * (1.0 - mask)

        prev_output_tokens = torch.cat((input_ids[:, :1], input_ids[:, 1:]), dim=1)

        input_ids = input_ids.to(torch.int64)
        prev_output_tokens = prev_output_tokens.to(torch.int64)
        tgt = input_ids.clone()
             
        model_cfg=dict(
            type="EDD",
            encoder=dict(
                type='RecurrentEncoder',
                mode="LSTM",
                embedding_size=512,
                hidden_size=512,
                dropout=0.5,
                num_layers=1,
                vocab_size=self.cfg.embedding_length,
                pad_token_id=self.cfg.pad_token_id
            ),
            decoder=dict(
                type="RecurrentDecoder",
                mode="LSTM",
                hidden_dim=512,
                dropout_out=0.5,
                vocab_size=self.cfg.embedding_length,
                pad_token_id=self.cfg.pad_token_id),
            discriminator=dict(
                type='RecurrentDiscriminator',
                mode="GRU",
                embedding_size=256,
                hidden_size=512,
                dropout=0.5,
                num_layers=1,
                out_size=512,
                vocab_size=self.cfg.embedding_length,
                pad_token_id=self.cfg.pad_token_id
            ),
        )
        model = build_from_cfg(model_cfg, GENERATORS)
        
        input = torch.randint(0, self.cfg.embedding_length, (32, 80))
        state = model.forward_encoder(input)
        # if model_cfg["encoder"]["num_layers"] > 1:
        #     assert state.hidden[0].shape == (model_cfg["encoder"]["num_layers"], self.cfg.train.batch_size, model_cfg["encoder"]["embedding_size"])
        # else:
        #     assert state.hidden[0].shape == (self.cfg.train.batch_size, model_cfg["encoder"]["embedding_size"])

        decoder_out, _ = model.forward_decoder(prev_output_tokens, state=state)
        assert decoder_out.shape == (batch_size, max_length, self.cfg.embedding_length)
        
        discriminated_out, discriminated_tgt = model.discriminator(decoder_out, tgt)

        # GREEDY SEARCH
        input_decoder = [[2]] * batch_size
        state = model.forward_encoder(
            input_ids,
        )
        state_list = state.as_list()

        seqs = torch.zeros((batch_size, max_length))
        for t in range(max_length):
            words, _, _ = model._decode_step(
                input_decoder, state_list,
                k=1,
                feed_all_timesteps=True,
            )
            seqs[:, t] = words[:, 0]
        
        print(words.shape)
        print(seqs.shape)
        
        assert False

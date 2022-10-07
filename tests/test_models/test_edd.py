import unittest

import torch
from catbird.utils import Config
from catbird.models.losses import pair_wise_loss
from catbird.models import build_generator_model
import torch.nn.functional as F


class TestEDD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile("configs/edd_quora.py")

        cfg_dict = dict(
            model=dict(
                type="EDD",
                encoder=dict(
                    type="RecurrentEncoder",
                    mode="LSTM",
                    embedding_size=512,
                    hidden_size=512,
                    dropout_in=0.5,
                    num_layers=1,
                ),
                decoder=dict(
                    type="RecurrentDecoder",
                    mode="LSTM",
                    hidden_size=512,
                    dropout_out=0.5,
                    num_layers=1,
                ),
                discriminator=dict(
                    type="RecurrentDiscriminator",
                    mode="GRU",
                    embedding_size=256,
                    hidden_size=512,
                    dropout=0.5,
                    num_layers=1,
                    out_size=512,
                ),
            )
        )
        cls.cfg = Config(cfg_dict)

        cls.cfg.embedding_length = 10000
        cls.cfg.pad_token_id = 0
        cls.cfg.decoder_start_token_id = 1

        cls.batch_size = 32
        cls.max_length = 80

        cls.model = build_generator_model(cls.cfg)

        assert type(cls.model).__name__ == "EDD"

        cls.device = next(cls.model.parameters()).device

        input_ids = torch.randint(1, 100, (cls.batch_size, cls.max_length))
        src_lengths = torch.randint(5, 15, (cls.batch_size,))

        mask = torch.zeros(input_ids.shape[0], input_ids.shape[1])
        mask[(torch.arange(input_ids.shape[0]), src_lengths)] = 1
        mask = mask.cumsum(dim=1)

        input_ids = input_ids * (1.0 - mask)
        input_ids = input_ids.to(torch.int64).to(cls.device)
        cls.input_ids = input_ids

    def test_generation(self):
        y_pred = self.model.generate(self.input_ids, max_length=self.max_length)

        assert y_pred.shape == (self.batch_size, self.max_length)

    def test_structure(self):
        encoder_outputs = self.model.encoder(input_ids=self.input_ids)

        # logits - [80, 32, 256]
        assert (
            self.max_length,
            self.batch_size,
            self.cfg.model.encoder.hidden_size,
        ) == encoder_outputs[0].shape

        # hidden state - [2, 32, 256]
        assert (
            (
                self.cfg.model.encoder.num_layers,
                self.batch_size,
                self.cfg.model.encoder.hidden_size,
            )
            == encoder_outputs[1][0].shape
            == encoder_outputs[1][1].shape
        )

        decoder_outputs = self.model.decoder(
            input_ids=self.input_ids, encoder_hidden_states=encoder_outputs[1]
        )

        assert len(decoder_outputs) == 2  # outs and hiddens
        assert (
            self.batch_size,
            self.max_length,
            self.cfg.embedding_length,
        ) == decoder_outputs[0].shape

        assert (
            (
                self.cfg.model.decoder.num_layers,
                self.batch_size,
                self.cfg.model.decoder.hidden_size,
            )
            == decoder_outputs[1][0].shape
            == decoder_outputs[1][1].shape
        )

        discriminated_out, discriminated_tgt = self.model.discriminator(
            decoder_outputs[0], self.input_ids
        )  # [32, 512]

        assert (
            (self.batch_size, self.cfg.model.discriminator.hidden_size)
            == discriminated_out.shape
            == discriminated_tgt.shape
        )

        pwloss = pair_wise_loss(discriminated_out, discriminated_tgt)

        assert torch.is_floating_point(pwloss)

        outputs = self.model(input_ids=self.input_ids, labels=self.input_ids)

        assert outputs[1].shape == (
            self.batch_size,
            self.max_length,
            self.cfg.embedding_length,
        )

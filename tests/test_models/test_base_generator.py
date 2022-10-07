import torch
import unittest
from catbird.utils import Config

# from catbird.models.generators.base import shift_right
from catbird.models import build_generator_model


class TestBaseGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ##### Config #####
        cfg_dict = dict(
            model=dict(
                type="StackedResidualLSTM",
                encoder=dict(
                    type="RecurrentEncoder",
                    mode="LSTM",
                    num_layers=2,
                    hidden_size=256,
                    dropout_in=0.5,
                ),
                decoder=dict(
                    type="RecurrentDecoder",
                    mode="LSTM",
                    num_layers=2,
                    hidden_size=256,
                    dropout_in=0.5,
                ),
            )
        )
        cls.cfg = Config(cfg_dict)

        cls.cfg.embedding_length = 10000
        cls.cfg.pad_token_id = 0
        cls.cfg.decoder_start_token_id = 1

        cls.batch_size = 32
        cls.max_length = 80

        ##### Model #####
        cls.model = build_generator_model(cls.cfg)

        cls.device = next(cls.model.parameters()).device

        ##### Input #####
        input_ids = torch.randint(1, 100, (cls.batch_size, cls.max_length))
        src_lengths = torch.randint(5, 15, (cls.batch_size,))

        mask = torch.zeros(input_ids.shape[0], input_ids.shape[1])
        mask[(torch.arange(input_ids.shape[0]), src_lengths)] = 1
        mask = mask.cumsum(dim=1)

        input_ids = input_ids * (1.0 - mask)
        input_ids = input_ids.to(torch.int64).to(cls.device)
        cls.input_ids = input_ids

    # def test_shift_right(self):
    #     cfg_dict = dict(
    #         num_workers=4,
    #         data=dict(
    #             max_length=40,
    #             train=dict(dataset_length=-1),
    #             val=dict(dataset_length=2000),
    #         ),
    #         train=dict(batch_size=32),
    #         model=dict(name="t5-small"),
    #     )
    #     cfg = Config(cfg_dict)

    #     tokenizer = build_tokenizer(cfg)
    #     cfg.embedding_length = len(tokenizer)
    #     cfg.pad_token_id = tokenizer.pad_token_id
    #     cfg.eos_token_id = (
    #         tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
    #     )
    #     cfg.decoder_start_token_id = (
    #         tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    #     )

    #     # 'What is ASMR? Does everyone experience it?</s><pad>'
    #     # 'What are some good tips to lose weight?</s><pad>'
    #     sample_inputs = torch.tensor(
    #         [
    #             [
    #                 363,
    #                 19,
    #                 6157,
    #                 9320,
    #                 58,
    #                 3520,
    #                 921,
    #                 351,
    #                 34,
    #                 58,
    #                 1,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #             ],
    #             [
    #                 363,
    #                 33,
    #                 128,
    #                 207,
    #                 2316,
    #                 12,
    #                 2615,
    #                 1293,
    #                 58,
    #                 1,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #                 0,
    #             ],
    #         ]
    #     )

    #     # '</s> What is ASMR? Does everyone experience it?</s><pad>'
    #     # '</s> What are some good tips to lose weight?</s><pad>'
    #     shifted_sample_inputs = shift_right(sample_inputs, cfg.decoder_start_token_id)

    #     assert torch.all(
    #         shifted_sample_inputs.eq(
    #             torch.cat(
    #                 (
    #                     torch.full(
    #                         (sample_inputs.size(0), 1), cfg.decoder_start_token_id
    #                     ),
    #                     sample_inputs[..., :-1].clone(),
    #                 ),
    #                 dim=1,
    #             )
    #         )
    #     )

    def test_encoder(self):
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

    def test_decoder(self):
        hidden_states = (
            torch.rand(2, 32, 256).to(self.device),
            torch.rand(2, 32, 256).to(self.device),
        )

        decoder_outputs = self.model.decoder(
            input_ids=self.input_ids, hidden_states=hidden_states
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

import unittest
import torch
from catbird.core import Config
from catbird.models import build_generator_model



class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ##### Config #####
        cfg_dict = dict(
            model = dict(
                type="StackedResidualLSTM",
                encoder=dict(
                    type="RecurrentEncoder",
                    mode="LSTM",
                    num_layers=2,
                    hidden_size=256,
                    dropout=0.5),
                decoder=dict(
                    type="RecurrentDecoder",
                    mode="LSTM",
                    num_layers=2,
                    hidden_size=256,
                    dropout_in=0.5),
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
        
        device = next(cls.model.parameters()).device
        
        ##### Input #####
        input_ids = torch.randint(1, 100, (cls.batch_size, cls.max_length))
        src_lengths = torch.randint(5, 15, (cls.batch_size,))

        mask = torch.zeros(input_ids.shape[0], input_ids.shape[1])
        mask[(torch.arange(input_ids.shape[0]), src_lengths)] = 1
        mask = mask.cumsum(dim=1)

        input_ids = input_ids * (1.0 - mask)
        input_ids = input_ids.to(torch.int64).to(device)
        cls.input_ids = input_ids
        


    def test_structure(self):  
        hidden_dim = 256
        num_layers = 2
        
        input_ids = self.input_ids.t()
        decoder_input_ids = input_ids[:-1, :]

        assert input_ids.shape == (self.max_length, self.batch_size)
        assert decoder_input_ids.shape == (self.max_length - 1, self.batch_size)

        out, hidden = self.model.encoder(input_ids)
        assert out.shape == (self.max_length, self.batch_size, hidden_dim)
        assert hidden.shape == (num_layers * 2, self.batch_size, hidden_dim)

        hidden = hidden[: self.model.decoder.num_layers]
        output = decoder_input_ids[0, :]
        output, hidden, attn_weights = self.model.decoder(output, hidden, out)
        assert output.shape == (self.batch_size, self.cfg.embedding_length)
        assert attn_weights.shape == (self.batch_size, 1, self.max_length)

        _, logits = self.model(input_ids.t(), input_ids.t())

        assert logits.shape == (
            self.max_length - 1,
            self.batch_size,
            self.cfg.embedding_length,
        )


    def test_generation(self):
                
        y_pred = self.model.generate(self.input_ids)
        
        assert y_pred.shape == (self.max_length, self.batch_size)
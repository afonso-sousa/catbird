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
                type="VanillaTransformer",
                encoder=dict(
                    type="TransformerEncoder",
                    embedding_size=256,
                    num_heads=8,
                    num_layers=3,
                    ffnn_size=512,
                    dropout=0.1,
                ),
                decoder=dict(
                    type="TransformerDecoder",
                    embedding_size=256,
                    num_heads=8,
                    num_layers=3,
                    ffnn_size=512,
                    dropout=0.1,
                ),
            ),
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
        
        input_ids = self.input_ids.t()
        decoder_input_ids = input_ids[:-1, :]

        assert input_ids.shape == (self.max_length, self.batch_size)
        assert decoder_input_ids.shape == (self.max_length - 1, self.batch_size)

        memory = self.model.encoder(input_ids)
        assert memory.shape == (self.max_length, self.batch_size, hidden_dim)

        decoder_out = self.model.decoder(decoder_input_ids, memory)
        assert decoder_out.shape == (self.max_length - 1, self.batch_size, self.cfg.embedding_length)

        _, logits = self.model(input_ids.t(), input_ids.t())

        assert logits.shape == (
            self.max_length - 1,
            self.batch_size,
            self.cfg.embedding_length,
        )


    def test_generation(self):
                
        y_pred = self.model.generate(self.input_ids)
        
        assert y_pred.shape == (self.max_length, 1)
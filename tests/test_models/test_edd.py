import unittest

import torch
import torch.nn as nn
from catbird.core import Config
from catbird.datasets import build_dataset, get_dataloader
from catbird.models.edd import EDD
from catbird.models.losses import sent_emb_loss
from catbird.models.utils import one_hot
from catbird.tokenizers import build_tokenizer

op = {'emb_dim': 512,
        'emb_hid_dim': 256,
        'enc_dropout': 0.5,
        'enc_rnn_dim': 512,
        'enc_dim': 512,
        'gen_rnn_dim': 512,
        'gen_dropout': 0.5}

class ParaphraseGenerator(nn.Module):
    """
    pytorch module which generates paraphrase of given phrase
    """
    def __init__(self, op):

        super(ParaphraseGenerator, self).__init__()

        # encoder :
        self.emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0))
        self.enc_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.enc_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # generator :
        self.gen_emb = nn.Embedding(op["vocab_sz"], op["emb_dim"])
        self.gen_rnn = nn.LSTM(op["enc_dim"], op["gen_rnn_dim"])
        self.gen_lin = nn.Sequential(
            nn.Dropout(op["gen_dropout"]),
            nn.Linear(op["gen_rnn_dim"], op["vocab_sz"]),
            nn.LogSoftmax(dim=-1))
        
        # pair-wise discriminator :
        self.dis_emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0),
        )
        self.dis_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.dis_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # some useful constants :
        self.max_seq_len = op["max_seq_len"]
        self.vocab_sz = op["vocab_sz"]

    def forward(self, phrase, sim_phrase=None, train=False):
        """
        forward pass
        inputs :-
        phrase : given phrase , shape = (max sequence length, batch size)
        sim_phrase : (if train == True), shape = (max seq length, batch sz)
        train : if true teacher forcing is used to train the module
        outputs :-
        out : generated paraphrase, shape = (max sequence length, batch size, )
        enc_out : encoded generated paraphrase, shape=(batch size, enc_dim)
        enc_sim_phrase : encoded sim_phrase, shape=(batch size, enc_dim)
        """

        if sim_phrase is None:
            sim_phrase = phrase

        if train:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            emb_sim_phrase_gen = self.gen_emb(sim_phrase)
            out_rnn, _ = self.gen_rnn(
                torch.cat([enc_phrase, emb_sim_phrase_gen[:-1, :]], dim=0))
            out = self.gen_lin(out_rnn)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        else:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            words = []
            h = None
            for __ in range(self.max_seq_len):
                word, h = self.gen_rnn(enc_phrase, hx=h)
                word = self.gen_lin(word)
                words.append(word)
                word = torch.multinomial(torch.exp(word[0]), 1)
                word = word.t()
                # print(word.shape)
                enc_phrase = self.gen_emb(word)
            out = torch.cat(words, dim=0)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase



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

        assert enc_out.shape == enc_tgt.shape == (self.cfg.train.batch_size, self.cfg.model.emb_dim)
        
        out = self.model.generate(self.src_ids)

        assert out.shape == (
            2 * self.cfg.train.batch_size,
            self.cfg.data.max_length,
            self.cfg.embedding_length,
        )
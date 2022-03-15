"""File to train a paraphrase generator."""
import argparse
import warnings
from functools import partial
from pathlib import Path

import ignite.distributed as idist
from catbird.apis import create_evaluator, create_trainer
from catbird.core import (
    Config,
    build_lr_scheduler,
    build_optimizer,
    log_basic_info,
    log_metrics,
    mkdir_or_exist,
)
from catbird.datasets import build_dataset, get_dataloader
from catbird.models import build_generator_model
from catbird.tokenizers import build_tokenizer
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
import torch
from torch import nn


import torch
from torch import nn
from torch.nn import functional as F

from catbird.models.postprocessing.search import SequenceGenerator
from catbird.models.state import State
from typing import Optional
from torch.nn import CrossEntropyLoss
import random

warnings.filterwarnings("ignore")

device = idist.device()

def ids_to_clean_text(generated_ids, tokenizer):
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return list(map(str.strip, gen_text))

def parse_args():
    parser = argparse.ArgumentParser(description="Train a paraphrase generator")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--gpus", type=int, help="number of gpus to use")
    group_gpus.add_argument("--gpu-ids", type=int, nargs="+", help="ids of gpus to use")

    args = parser.parse_args()

    return args


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

from catbird.models.generators.base import Seq2Seq
from catbird.models.encoders import RecurrentEncoder
from catbird.models.decoders import RecurrentDecoder


def shift_right(input_ids, decoder_start_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


class Seq2Seq(nn.Module):
    def __init__(self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.encoder = encoder
        self.decoder = decoder

    

    def forward(self, input_ids,
                tgt=None,
                decoder_input_ids=None,
                return_loss=False,
                **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs:
            encoder_out = self.encoder(input_ids, src_lengths)
            return self.decoder(decoder_input_ids, encoder_out)
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            decoder_input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_outputs = self.encoder(input_ids, **kwargs)
        
        if (tgt is not None) and (decoder_input_ids is None):   
            decoder_input_ids = shift_right(tgt, self.decoder_start_token_id)
        
        if (tgt is None) and (decoder_input_ids is None):
            decoder_input_ids = input_ids
            
        encoder_hidden_states = encoder_outputs[1]
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs)
        
        if return_loss:
            return self.loss(decoder_outputs[0], tgt, ignore_index=kwargs.get("ignore_index", -100))
        else:
            return decoder_outputs[0]


    def forward_encoder(self, input_ids, **kwargs):
        return self.encoder(input_ids, **kwargs)

    def forward_decoder(self, decoder_input_ids, **kwargs):
        return self.decoder(decoder_input_ids, **kwargs)

    def loss(self, logits, tgt, ignore_index=-100):
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
        return loss

    def _decode_step(
        self,
        input_list,
        state_list,
        k=1,
        apply_log_softmax=False,
        get_attention=False,
    ):

        device = next(self.decoder.parameters()).device

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        # last_tokens = [inputs[-1:] for inputs in input_list]
        # last_tokens = [torch.tensor([inputs[-1]]).to(device) for inputs in input_list]
        # inputs = torch.stack(last_tokens).view(-1, 1)
        inputs = torch.stack([torch.tensor(inp).to(device) for inp in input_list])

        state = State().from_list(state_list)
        logits, new_state = self.forward_decoder(inputs, state=state)
        
        new_state = state

        # use only last prediction
        # logits = logits.select(1, -1).contiguous()
        logits = logits[:, -1, :] # .to(device)

        if apply_log_softmax:
            logprobs = F.log_softmax(logits, dim=-1)
        else:
            logprobs = logits
        logprobs, words = logprobs.topk(k, dim=-1)
        new_state_list = [new_state[i] for i in range(len(input_list))]
        return words, logprobs, new_state_list

    def generate(
        self,
        input_ids,
        num_beams=3,
        max_sequence_length=50,
        length_normalization_factor=0,
        get_attention=False,
    ):
        batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1
        # input_decoder = [[self.decoder_start_token_id]] * batch_size
        # state = self.forward_encoder(
        #     input_ids,
        # )
        # state_list = state.as_list()
        
        if num_beams > 1:
            generator = SequenceGenerator(
                decode_step=self._decode_step,
                eos_token_id = self.eos_token_id,
                beam_size=num_beams,
                max_sequence_length=max_sequence_length,
                length_normalization_factor=length_normalization_factor,
                device = next(self.decoder.parameters()).device
            )
            output = generator.beam_search(input_decoder, state_list)
        else:
            input_ids = torch.full(
                (batch_size, 1),
                self.decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            output = self.greedy_search(input_ids, max_sequence_length, self.eos_token_id, self.pad_token_id) 
            # generator = SequenceGenerator(
            #     decode_step=self._decode_step,
            #     max_sequence_length=max_sequence_length,
            #     device = next(self.decoder.parameters()).device
            # )
            # preds = generator.greedy_search(input_decoder, state_list)
        
        return output

    def greedy_search(self, input_ids, max_length, eos_token_id=None, pad_token_id=None):
        batch_size = input_ids.shape[0]
        
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        
        cur_len = 1
        while cur_len < max_length:
            outputs = self(input_ids=input_ids)
            next_token_logits = outputs[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
            
        return input_ids


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)

    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        hidden = input_ids.new_zeros((1, batch_size, self.hidden_size)).float()
        embedded = self.embedding(input_ids)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden



class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size=256):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        hidden = input_ids.new_zeros((1, batch_size, self.hidden_size)).float()
        output = self.embedding(input_ids)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden



def training():
    device = idist.device()
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = Path("./work_dirs", Path(args.config).stem)

    mkdir_or_exist(Path(cfg.work_dir).resolve())

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
    )
    cfg.decoder_start_token_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id
        else tokenizer.cls_token_id
        if tokenizer.cls_token_id
        else tokenizer.pad_token_id
    )

    train_dataset = build_dataset(cfg, "train", tokenizer)
    train_dataloader = get_dataloader(cfg, "train", train_dataset)

    # model = build_generator_model(cfg)
    encoder = EncoderRNN(cfg.embedding_length)
    decoder = DecoderRNN(cfg.embedding_length)
    model = Seq2Seq(cfg.pad_token_id, cfg.eos_token_id, cfg.decoder_start_token_id, encoder, decoder).to(device)

    optimizer = build_optimizer(model, cfg.optimizer)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        cfg.scheduler.peak_lr,
        cfg.scheduler.num_warmup_epochs,
        cfg.train.num_epochs,
        cfg.train.get("epoch_length", len(train_dataset)),
    )

    for epoch in range(cfg.train.num_epochs):
        model.train()
        if cfg.data.get("mask_pad_token", False):
            ignore_index = -100
        else:
            ignore_index = cfg.pad_token_id
            
        tloss = 0
        for batch in train_dataloader:

            if batch["tgt"].device != device:
                batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

            loss = model(**batch, return_loss=True, ignore_index=ignore_index)
            tloss += loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch: {epoch} - Loss: {tloss.item()}")
        
        src_ids = batch["input_ids"]
        tgt = batch["tgt"]
        tgt = torch.where(tgt != -100, tgt, cfg.pad_token_id)

        y_pred = model.generate(src_ids, num_beams=1)

        preds = ids_to_clean_text(y_pred, tokenizer)
        tgt_text = ids_to_clean_text(tgt, tokenizer)

        preds = [_preds.split() for _preds in preds]
        tgt_text = [[_tgt.split()] for _tgt in tgt_text]

        print(f'\n Preds: {" ".join(preds[0])} \n\n Target: {" ".join(tgt_text[0][0])} \n')
        print(f'\n Preds: {" ".join(preds[1])} \n\n Target: {" ".join(tgt_text[1][0])} \n')
        
        print("---------------------------------------")
        
        src_ids = next(iter(train_dataloader))["input_ids"]
        tgt = next(iter(train_dataloader))["tgt"]
        tgt = torch.where(tgt != -100, tgt, cfg.pad_token_id)

        y_pred = model.generate(src_ids, num_beams=1)

        preds = ids_to_clean_text(y_pred, tokenizer)
        tgt_text = ids_to_clean_text(tgt, tokenizer)

        preds = [_preds.split() for _preds in preds]
        tgt_text = [[_tgt.split()] for _tgt in tgt_text]

        print(f'\n Preds: {" ".join(preds[0])} \n\n Target: {" ".join(tgt_text[0][0])} \n')
        print(f'\n Preds: {" ".join(preds[1])} \n\n Target: {" ".join(tgt_text[1][0])} \n')

        print("=" * 20)

if __name__ == "__main__":
    training()

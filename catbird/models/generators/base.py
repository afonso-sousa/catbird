import torch
from torch import nn
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder
from ..postprocessing.search import SequenceGenerator
from ..state import State
from typing import Optional
from torch.nn import CrossEntropyLoss
import random


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
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    

    def forward(self,
                input_ids,
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
    
    # def forward(self, input_ids, tgt, return_loss=True, teacher_forcing_ratio=0.5, **kwargs):
    #     max_len = input_ids.size(1)
        
    #     state = self.encoder(input_ids, **kwargs)
    #     decoder_input_ids = shift_right(tgt, self.decoder_start_token_id)
    #     # decoder_out, state = self.decoder(decoder_input_ids, state=state, **kwargs)
    #     outputs = []
    #     out = decoder_input_ids[:, 0] # [batch_size]
    #     for t in range(max_len):
    #         out, state = self.decoder(out, state=state)
    #         outputs.append(out)
    #         is_teacher = random.random() < teacher_forcing_ratio
    #         top1 = out.data.max(1)[1]
    #         out = decoder_input_ids[:, t] if is_teacher else top1

    #     logits = torch.stack(outputs, dim=1)
        
    #     if return_loss:
    #         return self.loss(logits, tgt, ignore_index=kwargs.get("ignore_index", -100))
    #     else:
    #         return logits

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
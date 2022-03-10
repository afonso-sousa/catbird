import torch
from torch import nn
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder
from ..postprocessing.search import SequenceGenerator
from ..state import State
from typing import Optional
from torch.nn import CrossEntropyLoss


class Seq2Seq(nn.Module):
    def __init__(self, decoder_start_token_id, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.decoder_start_token_id = decoder_start_token_id
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def _shift_right(self, input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.decoder_start_token_id

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(self, input_ids, tgt, return_loss=True, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs:
            encoder_out = self.encoder(input_ids, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        state = self.encoder(input_ids, **kwargs)
        prev_output_tokens = self._shift_right(tgt)
        decoder_out, _ = self.decoder(prev_output_tokens, state=state, **kwargs)
        if return_loss:
            return self.loss(decoder_out, tgt, ignore_index=kwargs.get("ignore_index", -100))
        else:
            return decoder_out

    def forward_encoder(self, input_ids, **kwargs):
        return self.encoder(input_ids, **kwargs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

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
        args_dict={},
        k=1,
        keep_all_timesteps=False,
        time_multiply=1,
        apply_lsm=True,
        get_attention=False,
    ):

        device = next(self.decoder.parameters()).device

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        # last_tokens = [inputs[-1:] for inputs in input_list]
        last_tokens = [torch.tensor([inputs[-1]]).to(device) for inputs in input_list]
        inputs = torch.stack(last_tokens).view(-1, 1)

        state = State().from_list(state_list)
        decode_inputs = dict(get_attention=get_attention, **args_dict)
        if time_multiply > 1:
            decode_inputs["time_multiply"] = time_multiply
        logits, new_state = self.forward_decoder(inputs, state=state)

        if not keep_all_timesteps:
            # use only last prediction
            logits = logits.select(1, -1).contiguous()
        if apply_lsm:
            logprobs = F.log_softmax(logits, dim=-1)
        else:
            logprobs = logits
        logprobs, words = logprobs.topk(k, dim=-1)
        new_state_list = [new_state[i] for i in range(len(input_list))]
        return words, logprobs, new_state_list

    def generate(
        self,
        input_encoder,
        num_beams=3,
        max_sequence_length=50,
        length_normalization_factor=0,
        get_attention=False,
    ):
        batch_size = input_encoder.size(0) if input_encoder.dim() > 1 else 1
        input_decoder = [[self.decoder_start_token_id]] * batch_size
        state = self.forward_encoder(
            input_encoder,
        )
        state_list = state.as_list()
        
        if num_beams > 1:
            generator = SequenceGenerator(
                decode_step=self._decode_step,
                beam_size=num_beams,
                max_sequence_length=max_sequence_length,
                length_normalization_factor=length_normalization_factor,
            )
            preds = generator.beam_search(input_decoder, state_list)
        else:
            generator = SequenceGenerator(
                decode_step=self._decode_step,
                max_sequence_length=max_sequence_length,
            )
            preds = generator.greedy_search(input_decoder, state_list)
        
        return preds

import torch
from torch import nn
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder
from ..postprocessing.search import SequenceGenerator
from ..state import State


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs:
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        state = self.encoder(src_tokens, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, state=state, **kwargs)
        return decoder_out

    def forward_encoder(self, inputs, hidden=None):
        return self.encoder(inputs, hidden)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def _decode_step(
        self,
        input_list,
        state_list,
        args_dict={},
        k=1,
        feed_all_timesteps=False,
        keep_all_timesteps=False,
        time_offset=0,
        time_multiply=1,
        apply_lsm=True,
        remove_unknown=False,
        get_attention=False,
        device_ids=None,
    ):

        view_shape = (-1, 1)
        time_dim = 1
        device = next(self.decoder.parameters()).device

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        # last_tokens = [inputs[-1:] for inputs in input_list]
        last_tokens = [torch.tensor([inputs[-1]]).to(device) for inputs in input_list]
        inputs = torch.stack(last_tokens).view(*view_shape)

        states = State().from_list(state_list)
        decode_inputs = dict(get_attention=get_attention, **args_dict)
        if time_multiply > 1:
            decode_inputs["time_multiply"] = time_multiply
        logits, new_states = self.forward_decoder(inputs, states=states)

        if not keep_all_timesteps:
            # use only last prediction
            logits = logits.select(time_dim, -1).contiguous()
        if apply_lsm:
            logprobs = F.log_softmax(logits, dim=-1)
        else:
            logprobs = logits
        logprobs, words = logprobs.topk(k, dim=-1)
        new_states_list = [new_states[i] for i in range(len(input_list))]
        return words, logprobs, new_states_list

    def generate(
        self,
        input_encoder,
        input_decoder,
        beam_size=None,
        max_sequence_length=None,
        length_normalization_factor=0,
        get_attention=False,
        device_ids=None,
        autoregressive=True,
    ):
        state = self.forward_encoder(
            input_encoder,
        )
        state_list = state.as_list()
        params = dict(
            decode_step=self._decode_step,
            beam_size=beam_size,
            max_sequence_length=max_sequence_length,
            length_normalization_factor=length_normalization_factor,
        )
        if autoregressive:
            generator = SequenceGenerator(**params)
        return generator.beam_search(input_decoder, state_list)

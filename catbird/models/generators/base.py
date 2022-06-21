
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder
from ..postprocessing import GenerationMixin


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class EncoderDecoderBase(nn.Module, GenerationMixin):
    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(EncoderDecoderBase, self).__init__()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward(
        self, input_ids, labels=None, decoder_input_ids=None, **kwargs
    ):
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
                `(batch, labels_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, labels_len, vocab)`
                - a dictionary with any model-specific outputs
        """
             
        if (labels is not None) and (decoder_input_ids is None):
            decoder_input_ids = labels[:, :-1].contiguous()
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id) # replace possible -100 values in labels by `pad_token_id`

        if (labels is None) and (decoder_input_ids is None):
            decoder_input_ids = input_ids
        
        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)

        encoder_hidden_states = encoder_outputs[1]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )

        loss = None
        if labels is not None:
            logits = decoder_outputs[0]

            targets = labels[:, :-1].contiguous()
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs


    def get_encoder(self):
        return self.encoder


    def get_decoder(self):
        return self.decoder


    def loss(self, logits, labels, ignore_index=-100):
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.view(-1))
        return loss

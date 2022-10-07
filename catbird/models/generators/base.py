import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    return shifted_input_ids


class EncoderDecoderBase(nn.Module):
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
        self,
        input_ids,
        labels=None,
        decoder_input_ids=None,
        incremental_state=None,
        **kwargs,
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
            # print("Shifting labels for decoder input...")
            # decoder_input_ids = shift_tokens_right(
            #     labels, self.pad_token_id, self.decoder_start_token_id
            # )
            decoder_input_ids = labels[:, :-1].contiguous()

        if (labels is None) and (decoder_input_ids is None):
            decoder_input_ids = input_ids

        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_out=encoder_outputs,
            incremental_state=incremental_state,
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = decoder_outputs[0]
            loss = self.loss(logits, labels)

        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def loss(self, logits, labels):
        targets = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss

    def forward_decoder(
        self, input_ids, encoder_out, incremental_state, temperature=1.0
    ):
        decoder_out = self.decoder.forward(
            input_ids,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        attn = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]["attn"]
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )
        probs = F.log_softmax(decoder_out_tuple[0], dim=-1)
        probs = probs[:, -1, :]
        return probs, attn

    def generate(self, input_ids, max_length=50, temperature=1.0):
        incremental_state = {}

        batch_size = input_ids.shape[0]

        encoder_outs = self.encoder(input_ids)

        input_ids = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        # TODO unfinished sentences
        # TODO sampling generation
        while cur_len < max_length:

            lprobs, _ = self.forward_decoder(
                input_ids,
                encoder_outs,
                incremental_state,
                temperature,
            )

            next_token = torch.argmax(lprobs, dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

        return input_ids

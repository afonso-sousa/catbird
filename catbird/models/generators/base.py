import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder


# def shift_tokens_right(input_ids, pad_token_id):
#     """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
#     prev_output_tokens = input_ids.clone()
#     index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
#     prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
#     prev_output_tokens[:, 1:] = input_ids[:, :-1]
#     return prev_output_tokens


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

    def shift_tokens_right(self, input_ids: torch.Tensor, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        return shifted_input_ids

    def forward(
        self,
        input_ids,
        labels=None,
        decoder_input_ids=None,
        incremental_state=None,
        **kwargs,
    ):
        if (labels is not None) and (decoder_input_ids is None):
            # print("Shifting labels for decoder input...")
            decoder_input_ids = self.shift_tokens_right(
                labels, self.decoder_start_token_id
            )
            # decoder_input_ids = labels[:, :-1].contiguous()

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
        # targets = labels[:, 1:].contiguous()
        targets = labels
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

    def generate(
        self,
        input_ids,
        graph=None,
        num_beams=1,
        max_length=50,
        min_length=1,
        temperature=1.0,
        sampling=False,
        num_return_sequences=1,
        top_k=50,
        top_p=1,
    ):

        batch_size = input_ids.shape[0]

        encoder_outs = self.encoder(input_ids)

        graph_embeddings = None
        if graph:
            graph_embeddings = self.graph_encoder(
                graph, self.encoder.embed_tokens.weight
            )

        input_ids = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        cur_len = 1

        if sampling:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if num_beams > 1:
            assert num_beams == 1, "Beam Search not yet implemented"
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outs,
                graph_embeddings=graph_embeddings,
                attention_mask=None,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        sampling,
        temperature,
        top_k,
        top_p,
        batch_size,
        encoder_outputs,
        graph_embeddings=None,
        attention_mask=None,
    ):
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        incremental_state = {}  # starts empty but is changed inside decoder forward

        while cur_len < max_length:

            decoder_out = self.decoder.forward(
                input_ids,
                encoder_out=encoder_outputs,
                incremental_state=incremental_state,
                graph_embeddings=graph_embeddings,
            )

            next_token_logits = decoder_out[0][:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                batch_size=batch_size,
                num_beams=1,
            )

            # next_token = torch.argmax(lprobs, dim=-1)

            # input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            # cur_len = cur_len + 1

            if sampling:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p
                )
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if self.eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (self.pad_token_id) * (
                    1 - unfinished_sents
                )
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if self.eos_token_id is not None:
                eos_in_sents = tokens_to_add == self.eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                    eos_in_sents.long()
                ).bool()
                sent_lengths.masked_fill_(
                    is_sents_unfinished_and_token_to_add_is_eos, cur_len
                )
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        return input_ids

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        cur_len,
        min_length,
        max_length,
        batch_size,
        num_beams,
    ):
        # set eos token prob to zero if min_length is not reached
        if self.eos_token_id is not None and cur_len < min_length:
            scores[:, self.eos_token_id] = -float("inf")

        return scores


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

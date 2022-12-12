# Generation Mixin. Based on Huggingface's Generation Mixin
# (<https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py>).
import torch
from torch import Tensor
from torch.nn import functional as F


class GenerationMixin:
    def enforce_repetition_penalty_(
        self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty
    ):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)."""
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        cur_len,
        min_length,
        max_length,
        batch_size,
        num_beams,
        repetition_penalty,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if self.eos_token_id is not None and cur_len < min_length:
            scores[:, self.eos_token_id] = -float("inf")

        return scores

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
        repetition_penalty=1.0,
        length_penalty=1.0,
    ):

        batch_size = input_ids.shape[0]

        encoder_outputs = self.encoder(input_ids)

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

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )
            # attention_mask = attention_mask.unsqueeze(1).expand(
            #     batch_size, effective_batch_mult * num_beams, input_ids_len
            # )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            # attention_mask = attention_mask.contiguous().view(
            #     effective_batch_size * num_beams, input_ids_len
            # )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        # encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])
        encoder_outputs = self.encoder.reorder_encoder_out(
            encoder_outputs, expanded_batch_idxs
        )

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=self.decoder.vocab_size,
                encoder_outputs=encoder_outputs,
                graph_embeddings=graph_embeddings,
                attention_mask=None,
            )
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
                repetition_penalty=repetition_penalty,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
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
        repetition_penalty,
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
                repetition_penalty=repetition_penalty,
            )

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

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        sampling,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        graph_embeddings=None,
        attention_mask=None,
    ):
        """Generate sequences for each example with beam search."""

        incremental_state = {}  # starts empty but is changed inside decoder forward

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if sampling is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            decoder_out = self.decoder.forward(
                input_ids,
                encoder_out=encoder_outputs,
                incremental_state=incremental_state,
                graph_embeddings=graph_embeddings,
            )  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = decoder_out[0][
                :, -1, :
            ]  # (batch_size * num_beams, vocab_size)

            scores = F.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            assert scores.shape == (
                batch_size * num_beams,
                vocab_size,
            ), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if sampling:
                _scores = scores + beam_scores[:, None].expand_as(
                    scores
                )  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(
                    probs, num_samples=2 * num_beams
                )  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(
                    _scores, -1, next_tokens
                )  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(
                    next_scores, descending=True, dim=1
                )
                next_tokens = torch.gather(
                    next_tokens, -1, next_scores_indices
                )  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(
                    scores
                )  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(
                    next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

            assert (
                next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
            )

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(
                        num_beams
                    )
                    next_batch_beam.extend(
                        [(0, self.pad_token_id, 0)] * num_beams
                    )  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (self.eos_token_id is not None) and (
                        token_id.item() == self.eos_token_id
                    ):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = (
                            beam_token_rank >= num_beams
                        )
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(),
                            beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append(
                            (beam_token_score, token_id, effective_beam_id)
                        )

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (
                    batch_idx + 1
                ), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if self.eos_token_id is not None and all(
                (token_id % vocab_size).item() != self.eos_token_id
                for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams]
                    == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = (
            batch_size if sampling else batch_size * num_return_sequences
        )
        output_num_return_sequences_per_batch = 1 if sampling else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(
                self.pad_token_id
            )

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = self.eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = (
                torch.stack(best).type(torch.long).to(next(self.parameters()).device)
            )

        return decoded


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


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

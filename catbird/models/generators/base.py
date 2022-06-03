
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ..builder import build_decoder, build_encoder
from ..postprocessing import GenerationMixin


# def shift_tokens_right(input_ids, decoder_start_token_id):
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
#     shifted_input_ids[..., 0] = decoder_start_token_id

#     assert torch.all(
#         shifted_input_ids >= 0
#     ).item(), "Verify that `shifted_input_ids` has only positive values"

#     return shifted_input_ids

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Seq2Seq(nn.Module, GenerationMixin):
    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(Seq2Seq, self).__init__()
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
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)
            
            # decoder_input_ids = labels[:, :-1].contiguous()
            # decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id) # replace possible -100 values in labels by `pad_token_id`

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
            # labels = labels[:, 1:].contiguous()
            # loss = self.loss(logits, labels, ignore_index=kwargs.get("ignore_index", -100))
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    # def forward(self, input_ids, labels, return_loss=True, teacher_forcing_ratio=0.5, **kwargs):
    #     input_ids = input_ids.t()
    #     labels = labels.t()
        
    #     batch_size = labels.shape[1]
    #     max_len = labels.shape[0]
        
    #     trg_vocab_size = self.decoder.vocab_size
        
    #     device = next(self.decoder.parameters()).device
        
    #     #tensor to store decoder outputs
    #     outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

    #     encoder_outputs = self.encoder(input_ids, **kwargs)
    #     hidden = encoder_outputs[1]
        
    #     # decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)
    #     # decoder_out, state = self.decoder(decoder_input_ids, state=state, **kwargs)
    #     out = labels[0, :]
        
    #     # out = decoder_input_ids[:, 0] # [batch_size]
    #     for t in range(1, max_len):
    #         out, state = self.decoder(out, hidden)
    #         hidden = state.hidden
    #         # outputs.append(out)
    #         # is_teacher = random.random() < teacher_forcing_ratio
    #         # top1 = out.data.max(1)[1]
    #         # out = decoder_input_ids[:, t] if is_teacher else top1
    #         outputs[t] = out
            
    #         teacher_force = random.random() < teacher_forcing_ratio
            
    #         #get the highest predicted token from our predictions
    #         top1 = out.argmax(1) 
            
    #         out = labels[t] if teacher_force else top1

    #     # logits = torch.stack(outputs, dim=1)
    #     logits = outputs

    #     if return_loss:
    #         return self.loss(logits, labels, ignore_index=kwargs.get("ignore_index", -100))
    #     else:
    #         return logits

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def loss(self, logits, labels, ignore_index=-100):
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        # output = logits.reshape(-1, logits.size(-1))
        # target = labels.reshape(-1)
        # loss = loss_fct(output, target)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.view(-1))
        return loss

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    # def _decode_step(
    #     self,
    #     input_list,
    #     state_list,
    #     k=1,
    #     feed_all_timesteps=False,
    #     apply_log_softmax=True,
    #     get_attention=False,
    # ):
    #     device = next(self.decoder.parameters()).device
    #     view_shape = (-1, 1)
    #     # For recurrent models, the last input frame is all we care about,
    #     # use feed_all_timesteps whenever the whole input needs to be fed
    #     # last_tokens = [inputs[-1:] for inputs in input_list]
    #     # last_tokens = [torch.tensor([inputs[-1]]).to(device) for inputs in input_list]
    #     # inputs = torch.stack(last_tokens).view(-1, 1)
    #     # inputs = torch.stack([torch.tensor(inp).to(device) for inp in input_list])
    #     if feed_all_timesteps:
    #         inputs = [torch.tensor(inp, device=device, dtype=torch.long)
    #                     for inp in input_list]
    #         inputs = torch.stack(inputs)

    #     else:
    #         last_tokens = [inputs[-1] for inputs in input_list]
    #         inputs = torch.stack(last_tokens).view(*view_shape)

    #     state = State().from_list(state_list)
    #     encoder_hidden_states = state.hidden
    #     logits, new_state = self.decoder(inputs, encoder_hidden_states=encoder_hidden_states)

    #     new_state = state

    #     # use only last prediction
    #     # logits = logits.select(1, -1).contiguous()
    #     logits = logits[:, -1, :]  # .to(device)

    #     if apply_log_softmax:
    #         logprobs = F.log_softmax(logits, dim=-1)
    #     else:
    #         logprobs = logits
    #     logprobs, words = logprobs.topk(k, dim=-1)
    #     new_state_list = [new_state[i] for i in range(len(input_list))]
    #     return words, logprobs, new_state_list

    # def generate(
    #     self,
    #     input_ids,
    #     num_beams=3,
    #     max_sequence_length=50,
    #     length_normalization_factor=0,
    #     get_attention=False,
    # ):
    #     batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1
    #     # input_decoder = [[self.decoder_start_token_id]] * batch_size
    #     # state = self.forward_encoder(
    #     #     input_ids,
    #     # )
    #     # state_list = state.as_list()
    #     # input_ids = input_ids.t()
        
    #     # encoder_outputs = self.encoder(input_ids)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
    #     # hidden = encoder_outputs[1]
        
    #     # seq_len, batch_size = input_ids.size()
    #     # decoded_batch = torch.zeros((batch_size, seq_len))
    #     # decoder_input = torch.tensor(input_ids.data[0, :]).cuda()
    #     # for t in range(seq_len):
    #     #     # decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
    #     #     out, state = self.decoder(decoder_input, hidden)
    #     #     hidden = state.hidden

    #     #     _, topi = out.data.topk(1)
    #     #     topi = topi.view(-1)
    #     #     decoded_batch[:, t] = topi

    #     #     decoder_input = topi.detach().view(-1)

    #     # return decoded_batch.int()

    # def generate(
    #     self,
    #     input_ids,
    #     num_beams=3,
    #     max_sequence_length=50,
    #     length_normalization_factor=0,
    #     get_attention=False,
    # ):
    #     batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1

    #     input_ids = torch.full(
    #         (batch_size, 1),
    #         self.decoder_start_token_id,
    #         dtype=torch.long,
    #         device=next(self.parameters()).device,
    #     )
    #     output = self.greedy_search(
    #         input_ids, max_sequence_length, self.eos_token_id, self.pad_token_id
    #     )

    #     return output

    # def greedy_search(
    #     self, input_ids, max_length, eos_token_id=None, pad_token_id=None
    # ):
    #     batch_size = input_ids.shape[0]

    #     # length of generated sentences / unfinished sentences
    #     unfinished_sents = input_ids.new(batch_size).fill_(1)
    #     sent_lengths = input_ids.new(batch_size).fill_(max_length)

    #     cur_len = 1
    #     while cur_len < max_length:
    #         model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            
    #         outputs = self(input_ids=input_ids)
    #         next_token_logits = outputs[:, -1, :]

    #         next_token = torch.argmax(next_token_logits, dim=-1)

    #         # update generations and finished sentences
    #         if eos_token_id is not None:
    #             # pad finished sentences if eos_token_id exist
    #             tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (
    #                 1 - unfinished_sents
    #             )
    #         else:
    #             tokens_to_add = next_token

    #         # add token and increase length by one
    #         input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
    #         cur_len = cur_len + 1

    #         # stop when there is a </s> in each sentence, or if we exceed the maximul length
    #         if unfinished_sents.max() == 0:
    #             break

    #     return input_ids


    # def generate(self, input_ids, input_decoder, beam_size=None,
	# 			 max_sequence_length=50, length_normalization_factor=0.0, top_p=0, top_k = 0,
	# 			 get_attention=False, eos_id = 2):
    #     encoder_outputs = self.encoder(input_ids)
    #     state_list = State(outputs=encoder_outputs[0], hidden=encoder_outputs[1]).as_list()
    #     generator = SequenceGenerator(
	# 		decode_step=self._decode_step,
    #         eos_id = eos_id,
	# 		beam_size=beam_size,
	# 		max_sequence_length=max_sequence_length,
	# 		length_normalization_factor=length_normalization_factor)
    #     return generator.beam_search(input_decoder, state_list)

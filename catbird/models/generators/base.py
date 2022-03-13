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
    def __init__(self, eos_token_id, decoder_start_token_id, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    

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
        prev_output_tokens = shift_right(tgt, self.decoder_start_token_id)
        decoder_out, _ = self.decoder(prev_output_tokens, state=state, **kwargs)
        if return_loss:
            return self.loss(decoder_out, tgt, ignore_index=kwargs.get("ignore_index", -100))
        else:
            return decoder_out
    
    # def forward(self, input_ids, tgt, return_loss=True, teacher_forcing_ratio=0.5, **kwargs):
    #     max_len = input_ids.size(1)
        
    #     state = self.encoder(input_ids, **kwargs)
    #     prev_output_tokens = shift_right(tgt, self.decoder_start_token_id)
    #     # decoder_out, state = self.decoder(prev_output_tokens, state=state, **kwargs)
    #     outputs = []
    #     out = prev_output_tokens[:, 0] # [batch_size]
    #     for t in range(max_len):
    #         out, state = self.decoder(out, state=state)
    #         outputs.append(out)
    #         is_teacher = random.random() < teacher_forcing_ratio
    #         top1 = out.data.max(1)[1]
    #         out = prev_output_tokens[:, t] if is_teacher else top1

    #     logits = torch.stack(outputs, dim=1)
        
    #     if return_loss:
    #         return self.loss(logits, tgt, ignore_index=kwargs.get("ignore_index", -100))
    #     else:
    #         return logits

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
                eos_token_id = self.eos_token_id,
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





#     def generate(
#         self,
#         input_ids,
#         num_beams=3,
#         max_sequence_length=50,
#     ):
#         batch_size, seq_len = input_ids.size()
#         device = next(self.decoder.parameters()).device
#         input_decoder = torch.tensor([self.decoder_start_token_id] * batch_size).to(device)
#         state = self.forward_encoder(
#             input_ids,
#         )
#         generator = Generator(
#             decode_step=self.decoder,
#             max_sequence_length=max_sequence_length)
#         if num_beams > 1:
#             pass
#         else:
#             preds = generator.greedy_search(input_decoder, state).int()

#         return preds
        
        
        
# # def decode(self, src, trg, method='beam-search'):
# #     encoder_output, hidden = self.encoder(src)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
# #     hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
# #     if method == 'beam-search':
# #         return self.beam_decode(trg, hidden, encoder_output)
# #     else:
# #         return self.greedy_search(trg, hidden, encoder_output)

# class Generator:
#     def __init__(
#         self,
#         decode_step,
#         eos_token_id=2,
#         beam_size=3,
#         max_sequence_length=50,
#     ):
#         self.decode_step = decode_step
#         self.eos_token_id = eos_token_id
#         self.beam_size = beam_size
#         self.max_sequence_length = max_sequence_length
        

#     def greedy_search(self, input_decoder, state):
#         '''
#         :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#         :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
#         :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#         :return: decoded_batch
#         '''
#         batch_size = len(input_decoder)
#         decoded_batch = torch.zeros((batch_size, self.max_sequence_length))
#         for t in range(self.max_sequence_length):
#             decoder_output, _ = self.decode_step(input_decoder, state=state)

#             _, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
#             topi = topi.view(-1)
#             decoded_batch[:, t] = topi

#             input_decoder = topi.detach().view(-1)

#         return decoded_batch


# def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
#     '''
#     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#     :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
#     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#     :return: decoded_batch
#     '''
#     target_tensor = target_tensor.permute(1, 0)
#     beam_width = 10
#     topk = 1  # how many sentence do you want to generate
#     decoded_batch = []

#     # decoding goes sentence by sentence
#     for idx in range(target_tensor.size(0)):  # batch_size
#         if isinstance(decoder_hiddens, tuple):  # LSTM case
#             decoder_hidden = (
#                 decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
#         else:
#             decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
#         encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]

#         # Start with the start of the sentence token
#         decoder_input = torch.LongTensor([SOS_token]).cuda()

#         # Number of sentence to generate
#         endnodes = []
#         number_required = min((topk + 1), topk - len(endnodes))

#         # starting node -  hidden vector, previous node, word id, logp, length
#         node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
#         nodes = PriorityQueue()

#         # start the queue
#         nodes.put((-node.eval(), node))
#         qsize = 1

#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             if qsize > 2000: break

#             # fetch the best node
#             score, n = nodes.get()
#             # print('--best node seqs len {} '.format(n.leng))
#             decoder_input = n.wordid
#             decoder_hidden = n.h

#             if n.wordid.item() == EOS_token and n.prevNode != None:
#                 endnodes.append((score, n))
#                 # if we reached maximum # of sentences required
#                 if len(endnodes) >= number_required:
#                     break
#                 else:
#                     continue

#             # decode for one step using decoder
#             decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)

#             # PUT HERE REAL BEAM SEARCH OF TOP
#             log_prob, indexes = torch.topk(decoder_output, beam_width)
#             nextnodes = []

#             for new_k in range(beam_width):
#                 decoded_t = indexes[0][new_k].view(-1)
#                 log_p = log_prob[0][new_k].item()

#                 node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
#                 score = -node.eval()
#                 nextnodes.append((score, node))

#             # put them into queue
#             for i in range(len(nextnodes)):
#                 score, nn = nextnodes[i]
#                 nodes.put((score, nn))
#                 # increase qsize
#             qsize += len(nextnodes) - 1

#         # choose nbest paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(topk)]

#         utterances = []
#         for score, n in sorted(endnodes, key=operator.itemgetter(0)):
#             utterance = []
#             utterance.append(n.wordid)
#             # back trace
#             while n.prevNode != None:
#                 n = n.prevNode
#                 utterance.append(n.wordid)

#             utterance = utterance[::-1]
#             utterances.append(utterance)

#         decoded_batch.append(utterances)

#     return decoded_batch


# class BeamSearchNode(object):
#     def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
#         '''
#         :param hiddenstate:
#         :param previousNode:
#         :param wordId:
#         :param logProb:
#         :param length:
#         '''
#         self.h = hiddenstate
#         self.prevNode = previousNode
#         self.wordid = wordId
#         self.logp = logProb
#         self.leng = length

#     def eval(self, alpha=1.0):
#         reward = 0
#         # Add here a function for shaping a reward
#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

#     def __lt__(self, other):
#         return self.leng < other.leng

#     def __gt__(self, other):
#         return self.leng > other.leng
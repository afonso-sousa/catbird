import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from beam_search import Beam

from .decoders.beam_search import Decoder
from .encoders.rvae_encoder import Encoder


class TDNN(nn.Module):
    def __init__(self, params):
        super(TDNN, self).__init__()
        self.params = params
        self.kernels = [
            Parameter(
                torch.Tensor(out_dim, self.params.char_embed_size, kW).uniform_(-1, 1)
            )
            for kW, out_dim in params.kernels
        ]
        self._add_to_parameters(self.kernels, "TDNN_kernel")

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]
        :return: tensor with shape [batch_size, max_seq_len, depth_sum]
        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert (
            input_size_len == 4
        ), "Wrong input rang, must be equal to 4, but {} found".format(input_size_len)

        [batch_size, seq_len, _, embed_size] = input_size

        assert (
            embed_size == self.params.char_embed_size
        ), "Wrong embedding size, must be equal to {}, but {} found".format(
            self.params.char_embed_size, embed_size
        )

        # leaps with shape
        x = (
            x.view(-1, self.params.max_word_len, self.params.char_embed_size)
            .transpose(1, 2)
            .contiguous()
        )

        xs = [F.tanh(F.conv1d(x, kernel)) for kernel in self.kernels]
        xs = [x.max(2, keepdim=True)[0].squeeze(2) for x in xs]

        x = torch.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name="{}-{}".format(name, i), param=parameter)


class Embedding(nn.Module):
    def __init__(self, params, path="../../../", flag=False):
        super(Embedding, self).__init__()

        self.params = params

        if flag == True:
            word_embed = np.load(path + "data/super/word_embeddings.npy")
        else:
            word_embed = np.load(path + "data/word_embeddings.npy")

        self.word_embed = nn.Embedding(
            self.params.word_vocab_size, self.params.word_embed_size
        )
        self.char_embed = nn.Embedding(
            self.params.char_vocab_size, self.params.char_embed_size
        )
        self.word_embed.weight = Parameter(
            torch.from_numpy(word_embed).float(), requires_grad=False
        )
        self.char_embed.weight = Parameter(
            torch.Tensor(
                self.params.char_vocab_size, self.params.char_embed_size
            ).uniform_(-1, 1)
        )

        self.TDNN = TDNN(self.params)

    def forward(self, word_input, character_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :param character_input: [batch_size, seq_len, max_word_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size + sum_depth]
        """

        assert (
            word_input.size()[:2] == character_input.size()[:2]
        ), "Word input and character input must have the same sizes, but {} and {} found".format(
            word_input.size(), character_input.size()
        )

        [batch_size, seq_len] = word_input.size()

        word_input = self.word_embed(word_input)

        character_input = character_input.view(-1, self.params.max_word_len)
        character_input = self.char_embed(character_input)
        character_input = character_input.view(
            batch_size, seq_len, self.params.max_word_len, self.params.char_embed_size
        )

        character_input = self.TDNN(character_input)

        result = torch.cat([word_input, character_input], 2)

        return result


def kld_coef(i):
    import math

    return (math.tanh((i - 3500) / 1000) + 1) / 2


class RVAE(nn.Module):
    def __init__(self, params, params_2):
        super(RVAE, self).__init__()

        self.params = params
        self.params_2 = params_2  # Encoder-2 parameters

        self.embedding = Embedding(self.params, "")
        self.embedding_2 = Embedding(self.params_2, "", True)

        self.encoder = Encoder(self.params)
        self.encoder_2 = Encoder(self.params_2)

        self.context_to_mu = nn.Linear(
            self.params.encoder_rnn_size * 2, self.params.latent_variable_size
        )
        self.context_to_logvar = nn.Linear(
            self.params.encoder_rnn_size * 2, self.params.latent_variable_size
        )

        self.decoder = Decoder(self.params_2)  # change this to params_2

    def forward(
        self,
        drop_prob,
        encoder_word_input=None,
        encoder_character_input=None,
        encoder_word_input_2=None,
        encoder_character_input_2=None,
        decoder_word_input_2=None,
        decoder_character_input_2=None,
        z=None,
        initial_state=None,
    ):

        # Modified the parameters of forward function according to Encoder-2
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param z: context if sampling is performing
        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        use_cuda = self.embedding.word_embed.weight.is_cuda

        if z is None:
            """ Get context from encoder and sample z ~ N(mu, std)
            """
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            """ ===================================================Doing the same for encoder-2===================================================
            """
            [batch_size_2, _] = encoder_word_input_2.size()

            encoder_input_2 = self.embedding_2(
                encoder_word_input_2, encoder_character_input_2
            )

            """ ==================================================================================================================================
            """

            context, h_0, c_0 = self.encoder(encoder_input, None)

            State = (h_0, c_0)  # Final state of Encoder-1
            context_2, _, _ = self.encoder_2(
                encoder_input_2, State
            )  # Encoder_2 for Ques_2

            mu = self.context_to_mu(context_2)
            logvar = self.context_to_logvar(context_2)
            std = torch.exp(0.5 * logvar)

            z = Variable(torch.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (
                (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1))
                .mean()
                .squeeze()
            )

            # encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            # _ , h_0 , c_0 = self.encoder_3(encoder_input, None)
            initial_state = State  # Final state of Encoder-1

        else:
            kld = None
            mu = None
            std = None

        decoder_input_2 = self.embedding_2.word_embed(
            decoder_word_input_2
        )  # What to do with this decoder input ? --> Slightly resolved
        out, final_state = self.decoder(
            decoder_input_2, z, drop_prob, initial_state
        )  # Take a look at the decoder

        return out, final_state, kld, mu, std

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader, batch_loader_2):
        def train(i, batch_size, use_cuda, dropout, start_index):
            input = batch_loader.next_batch(batch_size, "train", start_index)
            input = [Variable(torch.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [
                encoder_word_input,
                encoder_character_input,
                _,
                _,
                target,
            ] = input

            """ =================================================== Input for Encoder-2 ========================================================
            """

            input_2 = batch_loader_2.next_batch(batch_size, "train", start_index)
            input_2 = [Variable(torch.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]

            [
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                target,
            ] = input_2

            """ ================================================================================================================================
            """
            # exit()

            logits, _, kld, _, _ = self(
                dropout,
                encoder_word_input,
                encoder_character_input,
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                z=None,
            )

            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i)

        return train

    def validater(self, batch_loader, batch_loader_2):
        def validate(batch_size, use_cuda, start_index):
            input = batch_loader.next_batch(batch_size, "valid", start_index)
            input = [Variable(torch.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [
                encoder_word_input,
                encoder_character_input,
                decoder_word_input,
                decoder_character_input,
                target,
            ] = input

            """ ==================================================== Input for Encoder-2 ========================================================
            """

            input_2 = batch_loader_2.next_batch(batch_size, "valid", start_index)
            input_2 = [Variable(torch.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]
            [
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                target,
            ] = input_2

            """ ==================================================================================================================================
            """

            logits, _, kld, _, _ = self(
                0.0,
                encoder_word_input,
                encoder_character_input,
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                z=None,
            )

            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda, State):
        # seed = Variable(torch.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(
            torch.from_numpy(decoder_character_input_np).long()
        )

        if use_cuda:
            decoder_word_input, decoder_character_input = (
                decoder_word_input.cuda(),
                decoder_character_input.cuda(),
            )

        result = ""

        initial_state = State

        for i in range(seq_len):
            logits, initial_state, _, _, _ = self(
                0.0,
                None,
                None,
                None,
                None,
                decoder_word_input,
                decoder_character_input,
                seed,
                initial_state,
            )

            # forward(self, drop_prob,
            #           encoder_word_input=None, encoder_character_input=None,
            #           encoder_word_input_2=None, encoder_character_input_2=None,
            #           decoder_word_input_2=None, decoder_character_input_2=None,
            #           z=None, initial_state=None):

            # logits = logits.view(-1, self.params.word_vocab_size)
            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            # print '---------------------------------------'
            # print 'Printing logits'
            # print logits
            # print '------------------------------------------'

            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(
                prediction.data.cpu().numpy()[-1]
            )

            if word == batch_loader.end_token:
                break

            result += " " + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array(
                [[batch_loader.encode_characters(word)]]
            )

            decoder_word_input = Variable(
                torch.from_numpy(decoder_word_input_np).long()
            )
            decoder_character_input = Variable(
                torch.from_numpy(decoder_character_input_np).long()
            )

            if use_cuda:
                decoder_word_input, decoder_character_input = (
                    decoder_word_input.cuda(),
                    decoder_character_input.cuda(),
                )

        return result

    def sampler(
        self,
        batch_loader,
        batch_loader_2,
        seq_len,
        seed,
        use_cuda,
        i,
        beam_size,
        n_best,
    ):
        input = batch_loader.next_batch(1, "valid", i)
        input = [Variable(torch.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]
        [
            encoder_word_input,
            encoder_character_input,
            decoder_word_input,
            decoder_character_input,
            target,
        ] = input

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        _, h0, c0 = self.encoder(encoder_input, None)
        State = (h0, c0)

        # print '----------------------'
        # print 'Printing h0 ---------->'
        # print h0
        # print '----------------------'

        # State = None
        results, scores = self.sample_beam(
            batch_loader_2, seq_len, seed, use_cuda, State, beam_size, n_best
        )

        return results, scores

    def sample_beam(
        self, batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best
    ):
        # seed = Variable(torch.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(
            torch.from_numpy(decoder_character_input_np).long()
        )

        if use_cuda:
            decoder_word_input, decoder_character_input = (
                decoder_word_input.cuda(),
                decoder_character_input.cuda(),
            )

        dec_states = State

        # print '========= Before ================'
        # print "dec_states:", dec_states[0].size()
        # print "dec_states:", dec_states[1].size()
        # print '=================================='

        # dec_states = [
        #     Variable(dec_states[0].repeat(1, beam_size, 1)),
        #     Variable(dec_states[1].repeat(1, beam_size, 1))
        # ]
        dec_states = [
            dec_states[0].repeat(1, beam_size, 1),
            dec_states[1].repeat(1, beam_size, 1),
        ]

        # print'========== After =================='
        # print "dec_states:", dec_states[0].size()
        # print "dec_states:", dec_states[1].size()
        # print '=================================='
        # exit()

        drop_prob = 0.0
        beam_size = beam_size
        batch_size = 1

        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(seq_len):

            input = (
                torch.stack([b.get_current_state() for b in beam if not b.done])
                .t()
                .contiguous()
                .view(1, -1)
            )

            trg_emb = self.embedding_2.word_embed(Variable(input).transpose(1, 0))

            # print trg_emb.size()
            # print seed.size()

            trg_h, dec_states = self.decoder.only_decoder_beam(
                trg_emb, seed, drop_prob, dec_states
            )

            # trg_h, (trg_h_t, trg_c_t) = self.model.decoder(trg_emb, (dec_states[0].squeeze(0), dec_states[1].squeeze(0)), context )

            # print trg_h.size()
            # print trg_h_t.size()
            # print trg_c_t.size()

            # dec_states = (trg_h_t, trg_c_t)

            # print 'State dimension ----------->'
            # print State[0].size()
            # print State[1].size()
            # print '======================================='
            # print "dec_states:", dec_states[0].size()
            # print "dec_states:", dec_states[1].size()
            # print '========== Things successful ==========='

            # exit()

            dec_out = trg_h.squeeze(1)

            # print "dec_out:", dec_out.size()

            out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)

            word_lk = (
                out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()
            )

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(1, beam[b].get_current_origin())
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, self.params.decoder_rnn_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx).view(*new_size))

            dec_states = (update_active(dec_states[0]), update_active(dec_states[1]))
            dec_out = update_active(dec_out)
            # context = update_active(context)

            remaining_sents = len(active)

        # (4) package everything up

        allHyp, allScores = [], []

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            # print scores
            # print ks
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            # print hyps
            # print "------------------"
            allHyp += [hyps]

        # print '==== Complete ========='

        return allHyp, allScores


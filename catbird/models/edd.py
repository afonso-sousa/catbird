from logging import Logger
from typing import Tuple, Union

"""Class with facilities to train and evaluate a Encoder-Decoder-Discriminator model."""
import ignite.distributed as idist
import torch
from torch import nn
from torch import optim
from catbird.core import Config  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from .utils import freeze_params, one_hot
from .losses import sent_emb_loss


class GRUEncoder(nn.Module):
    """Simple GRU-based encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_out_dims: int,
        encoder_hidden_dims: int,
        encoder_out_dims: int,
        dropout_proba: int = 0.5,
    ):
        super(GRUEncoder, self).__init__()
        self.vocab_size = vocab_size

        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_out_dims)
        self.gru_encoder = nn.GRU(embedding_out_dims, encoder_hidden_dims)
        self.linear = nn.Sequential(
            nn.Dropout(dropout_proba), nn.Linear(encoder_hidden_dims, encoder_out_dims),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define model's forward pass.

        Args:
            input (torch.Tensor): Batch of sentences as id sequences.

        Returns:
            torch.Tensor: Output of last linear layer.
        """
        embedding_out = self.embedding_layer(input)
        
        # B x T x C -> T x B x C
        embedding_out = embedding_out.transpose(0, 1)
        
        encoder_out = self.gru_encoder(embedding_out)[1]
        out = self.linear(encoder_out)

        return out


class EDD(nn.Module):
    """
    LSTM-based Encoder-Decoder-Discriminator (EDD) architecture.

    This is an implementation of paper `Learning Semantic Sentence Embeddings
    using Sequential Pair-wise Discriminator <https://aclanthology.org/C18-1230/>`.
    """

    def __init__(self, cfg: Config) -> None:
        """Initialize Encoder-Decoder-Disciminator architecture.

        Args:
            cfg (Config): Config instance with configurations.
        """
        super(EDD, self).__init__()
        self.max_length = cfg.data.max_length
        self.vocab_size = cfg.embedding_length

        self.encoder = GRUEncoder(
            self.vocab_size,
            cfg.model.emb_dim,
            cfg.model.enc_rnn_dim,
            cfg.model.enc_dim,
            cfg.model.enc_dropout,
        )

        self.gen_emb = nn.Embedding(self.vocab_size, cfg.model.emb_dim)
        self.gen_rnn = nn.LSTM(cfg.model.enc_dim, cfg.model.gen_rnn_dim)
        self.generator_linear = nn.Sequential(
            nn.Dropout(cfg.model.gen_dropout),
            nn.Linear(cfg.model.gen_rnn_dim, self.vocab_size),
            nn.LogSoftmax(dim=-1),
        )

        # pair-wise discriminator :
        self.dis_emb_layer = nn.Sequential(
            nn.Linear(self.vocab_size, cfg.model.emb_hid_dim),
            nn.Threshold(0.000001, 0),
            nn.Linear(cfg.model.emb_hid_dim, cfg.model.emb_dim),
            nn.Threshold(0.000001, 0),
        )
        self.dis_rnn = nn.GRU(cfg.model.emb_dim, cfg.model.enc_rnn_dim)
        self.dis_lin = nn.Sequential(
            nn.Dropout(cfg.model.enc_dropout),
            nn.Linear(cfg.model.enc_rnn_dim, cfg.model.enc_dim),
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass method.

        Args:
            input (torch.Tensor): input batch (batch_size, max_length)
            target (torch.Tensor): target batch (batch_size, max_length)

        Returns:
            [Tuple[torch.Tensor]]: generated paraphrase. shape (batch size, max length, vocab length)
                                   encoded generated paraphrase, shape=(batch size, enc_dim)
                                   encoded target, shape=(batch size, enc_dim)

        """
        enc_input = self.encoder(input)
        
        # B x T x C -> T x B x C
        target = target.transpose(0, 1)

        # Generator
        embedding_out = self.gen_emb(target)
        hidden_state, _ = self.gen_rnn(
            torch.cat([enc_input, embedding_out[:-1, :]], dim=0)
        )
        out = self.generator_linear(hidden_state)

        # propagated from shared discriminator to calculate
        # pair-wise discriminator loss
        enc_target = self.dis_lin(
            self.dis_rnn(self.dis_emb_layer(one_hot(target, self.vocab_size)))[1]
        )
        enc_out = self.dis_lin(self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        out = out.transpose(0, 1)
        enc_out = enc_out.squeeze(0)
        enc_target = enc_target.squeeze(0)
        
        return out, enc_out, enc_target

    def generate(self, input, target=None):
        # batch_size = input.size(0)

        if target is None:
            target = input

        enc_input = self.encoder(input)

        # generate similar phrase using teacher forcing
        words = []
        for _ in range(self.max_length):
            word, _ = self.gen_rnn(enc_input)
            word = self.generator_linear(word)
            words.append(word)
            word = torch.multinomial(torch.exp(word[0]), 1)
            word = word.t()
            enc_input = self.gen_emb(word)
        out = torch.cat(words, dim=0).transpose(0, 1)

        return out

    @property
    def get_encoder(self):
        return self.encoder

def train_step(
    cfg: Config,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: Union[str, torch.device],
    scaler: GradScaler,
):
    """Decorate train step to add more parameters.

    Args:
        cfg (Config): Config instance with configurations.
        model (nn.Module): a Pytorch model.
        optimizer (optim.Optimizer): a Pytorch optimizer.
        device (Union[str, torch.device]): specifies which device updates are accumulated on.
        scaler (GradScaler): GradScaler instance for gradient scaling.
    """

    def routine(engine, batch):
        accumulation_steps = cfg.train.get("accumulation_steps", 1)
        with_amp = cfg.train.get("with_amp", False)

        if cfg.data.get("tokenizer", None):
            ignore_index = -100
        else:
            ignore_index = cfg.pad_token_id
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)

        model.train()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        tgt = batch["tgt"]

        with autocast(enabled=with_amp):
            out, enc_out, enc_target = model(src_ids, tgt)

            loss1 = loss_fct(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
            loss2 = sent_emb_loss(enc_out, enc_target)
            loss = loss1 + loss2

            loss /= accumulation_steps

        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    return routine


def evaluate_step(
    cfg: Config,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: Union[str, torch.device],
    logger: Logger,
):
    """Decorate evaluate step to add more parameters.

    Args:
        cfg (Config): Config instance with configurations.
        model (nn.Module): a Pytorch model.
        tokenizer (AutoTokenizer): a Pytorch optimizer.
        device (Union[str, torch.device]): specifies which device updates are accumulated on.
        logger (Logger): a Logger instance.
    """

    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return list(map(str.strip, gen_text))

    @torch.no_grad()
    def routine(engine, batch):
        model.eval()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        tgt = batch["tgt"]
        out = model.generate(src_ids)
        y_pred = torch.argmax(out, dim=-1)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(tgt)
        preds = [_preds.split() for _preds in preds]
        tgt = [[_tgt.split()] for _tgt in tgt]

        if engine.state.iteration % cfg.print_output_every == 0:
            logger.info(f'\n Preds : {" ".join(preds[0])} \n')
            logger.info(f'\n Target : {" ".join(tgt[0][0])} \n')
        return preds, tgt

    return routine


def initialize(cfg: Config) -> nn.Module:
    """Initialize T5 conditional generator based on the given configurations.

    Args:
        cfg (Config): configuration file

    Returns:
        Tuple[nn.Module, optim.Optimizer]: model and optimizer
    """
    model = EDD(cfg)

    # lr = cfg.train.learning_rate * idist.get_world_size()
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p
    #             for n, p in model.named_parameters()
    #             if not any(nd_entry in n for nd_entry in no_decay)
    #         ],
    #         "weight_decay": cfg.train.weight_decay,
    #     },
    #     {
    #         "params": [
    #             p
    #             for n, p in model.named_parameters()
    #             if any(nd_entry in n for nd_entry in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]
    if cfg.model.freeze_encoder:
        freeze_params(model.get_encoder())

    model = idist.auto_model(model)
    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = idist.auto_optim(optimizer)

    return model

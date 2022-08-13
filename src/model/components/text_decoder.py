"""Text decoders.

Input:
    features: tensor of shape (B, N, d1), local image features
    context: tensor of shape (B, N, d2)

Output:
    words: tensor of shape (B, N, L, d3), word distributions

Available context encoders:
    - lstm decoder
    - transformer decoder
"""
import logging
from typing import Any, Dict, Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.dataset.text import SimpleTokenizer

from .x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    Decoder,
    FixedPositionalEmbedding,
    always,
)

logger = logging.getLogger(__name__)

tokenizer = SimpleTokenizer()
N_WORDS = len(tokenizer)
PAD_IDX = tokenizer.pad_idx


class IndependentLSTMText(nn.Module):
    """Decode texts with independent LSTMs."""

    def __init__(
        self,
        state_dim: int = 512,
        feature_dim: int = 512,
        max_seq_len: int = 5,
        word_emb_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout=0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.state_project = (
            nn.Linear(state_dim, word_emb_dim)
            if state_dim != word_emb_dim
            else nn.Identity()
        )
        self.embed = nn.ModuleList(
            [
                nn.Embedding(N_WORDS, hidden_dim, PAD_IDX)
                for _ in range(self.max_seq_len)
            ]
        )
        self.feature_project = (
            nn.Linear(feature_dim, word_emb_dim)
            if feature_dim != word_emb_dim
            else nn.Identity()
        )
        self.input_dropout = nn.Dropout(dropout)
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=word_emb_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                )
                for _ in range(self.max_seq_len)
            ]
        )
        self.output_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, N_WORDS)

    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        captions: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        if captions is None:
            return self.inference(state, features)

        h0, c0 = state

        B, N, L = captions.size()
        assert mask.size() == captions.size()
        assert h0.size() == (self.num_layers, B, self.state_dim)
        assert c0.size() == (self.num_layers, B, self.state_dim)

        # prepare inputs
        h0 = self.state_project(h0)
        c0 = self.state_project(c0)
        features = self.feature_project(features)
        captions = torch.cat(
            [self.embed[i](captions[:, i : i + 1, :]) for i in range(N)], dim=1
        )
        caption_start = features.unsqueeze(dim=2)
        captions = torch.cat((caption_start, captions), dim=2)[:, :, :-1, :]

        # lstm forward pass
        lengths = mask.sum(dim=-1).cpu()
        outputs = []
        captions = self.input_dropout(captions)
        for i in range(N):
            input_unit = pack_padded_sequence(
                captions[:, i, :, :],
                lengths[:, i],
                batch_first=True,
                enforce_sorted=False,
            )
            output, _ = self.lstms[i](input_unit, (h0, c0))
            output, _ = pad_packed_sequence(
                output, batch_first=True, total_length=L
            )
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.output_dropout(outputs)

        # map into words
        outputs = self.linear(outputs)

        return outputs

    def inference(context: torch.Tensor):
        pass


class ContextLSTMText(nn.Module):
    """Implementation of the GLocal text decoder.

    Note: the inputs of LSTM is a concatenation of the word embedding and the
    glocal features.
    """

    def __init__(
        self,
        context_dim: int = 512,
        word_emb_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        embed_dropout=0.1,
        lstm_dropout=0.5,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(N_WORDS, hidden_dim, PAD_IDX)
        self.dropout_embed = nn.Dropout(p=embed_dropout)
        self.lstm_input_dim = word_emb_dim + context_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout_lstm = nn.Dropout(p=lstm_dropout)
        self.linear = nn.Linear(hidden_dim, N_WORDS)

    def forward(
        self,
        context: torch.Tensor,
        captions: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        if captions is None:
            return self.inference(context)

        B, N, L = captions.size()
        assert mask.size() == captions.size()

        # prepare inputs
        inputs = self.embed(captions)
        inputs = self.dropout_embed(inputs)
        context = repeat(
            context, "B N d -> B N L d", B=B, N=N, L=L, d=self.context_dim
        )
        inputs = torch.cat([inputs, context], dim=-1)

        # lstm forward pass
        inputs = rearrange(
            inputs, "B N L d -> (B N) L d", B=B, N=N, L=L, d=self.lstm_input_dim
        )
        lengths = rearrange(mask.sum(dim=-1), "B N -> (B N)", B=B, N=N).cpu()
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(inputs)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=L
        )

        # map into words
        output = self.dropout_lstm(output)
        output = self.linear(output)
        output = rearrange(
            output, "(B N) L d -> B N L d", B=B, N=N, L=L, d=N_WORDS
        )

        return output

    def inference(context: torch.Tensor):
        pass


class TransformerText(nn.Module):
    """Implementation of the Transformer text decoder."""

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 512,
        heads: int = 8,
        num_layers: int = 2,
        dropout=0.1,
        position_embedding: str = "relative",
        max_words: int = 20,
        decoder_args: Dict[str, Any] = {},
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.position_embedding = position_embedding
        self.max_words = max_words

        self.context_project = (
            nn.Linear(context_dim, hidden_dim)
            if context_dim != hidden_dim
            else nn.Identity()
        )

        self.embed = nn.Embedding(N_WORDS, hidden_dim, PAD_IDX)
        self.dropout_embed = nn.Dropout(p=dropout)

        self.pos_embed = self.setup_position_embedding()
        self.decoder = Decoder(
            dim=hidden_dim,
            depth=num_layers,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rel_pos_bias=self.rel_pos_bias,
            position_infused_attn=self.position_infused,
            **decoder_args,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_logits = nn.Linear(hidden_dim, N_WORDS)

    def setup_position_embedding(self):
        assert self.position_embedding in [
            "fixed",  # Fixed sinusoidal positional embedding
            "absolute",  # Learned positional encoding
            "infused_fixed",  # Shortformer: https://arxiv.org/pdf/2012.15832.pdf
            "relative",  # Simplified relative positional encoding from T5
        ]
        # these embeddings are injected to all layers
        self.position_infused = self.position_embedding == "infused_fixed"
        self.rel_pos_bias = self.position_embedding == "relative"

        # first layer only
        if self.position_embedding == "fixed":
            pos_embed = FixedPositionalEmbedding(self.hidden_dim)
        elif self.position_embedding == "absolute":
            pos_embed = AbsolutePositionalEmbedding(
                self.hidden_dim, self.max_words
            )
        else:
            pos_embed = always(0)
        return pos_embed

    def forward(
        self,
        context: torch.Tensor,
        captions: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        if captions is None:
            return self.inference(context)

        B, N, L = captions.size()

        # prepare inputs
        caption_start = self.context_project(context)
        caption_start = caption_start.unsqueeze(2)
        captions = self.embed(captions)
        captions = torch.cat((caption_start, captions), dim=2)[:, :, :-1, :]
        captions = rearrange(
            captions, "B N L d -> (B N) L d", B=B, N=N, L=L, d=self.hidden_dim
        )
        mask = rearrange(mask, "B N L -> (B N) L", B=B, N=N, L=L)

        # positional embedding
        captions = captions + self.pos_embed(captions)
        captions = self.dropout_embed(captions)

        # decoder forward pass
        output = self.decoder(captions, mask=mask)
        output = self.norm(output)

        # map into words
        logits = self.to_logits(output)
        logits = rearrange(
            logits, "(B N) L d -> B N L d", B=B, N=N, L=L, d=N_WORDS
        )

        return logits

    def inference(self, context: torch.Tensor):
        pass

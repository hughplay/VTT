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


def mask_select(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select tensor elements where mask is True.

    Args:
        tensor: tensor of shape (B, ?)
        mask: tensor of shape (B)

    Returns:
        tensor of shape (M, ?), where M is the number of True elements in mask
    """
    return tensor[mask]


def mask_unselect(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Unselect tensor elements where mask is True.

    Args:
        tensor: tensor of shape (M, ?)
        mask: tensor of shape (B)

    Returns:
        tensor of shape (B, ?), where False positions are filled with 0
    """
    tensor_size = list(tensor.size())
    assert tensor_size[0] == mask.sum().item()
    tensor_size[0] = mask.size(0)
    full_tensor = torch.zeros(
        tensor_size, dtype=tensor.dtype, device=tensor.device
    )
    full_tensor[mask] = tensor
    return full_tensor


class IndependentLSTMText(nn.Module):
    """Decode texts with independent LSTMs."""

    def __init__(
        self,
        state_dim: int = 512,
        feature_dim: int = 512,
        max_seq_len: int = 12,
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
            [nn.Embedding(N_WORDS, hidden_dim) for _ in range(self.max_seq_len)]
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
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        trans_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        if label_ids is None:
            return self.inference(lstm_state, features, trans_mask)

        h0, c0 = lstm_state

        B, N, L = label_ids.size()
        assert label_mask.size() == label_ids.size()
        assert h0.size() == (self.num_layers, B, self.state_dim)
        assert c0.size() == (self.num_layers, B, self.state_dim)

        # prepare inputs
        h0 = self.state_project(h0)
        c0 = self.state_project(c0)
        features = self.feature_project(features)
        labels = torch.cat(
            [self.embed[i](label_ids[:, i : i + 1, :]) for i in range(N)], dim=1
        )
        start_tok = features.unsqueeze(dim=2)
        labels = torch.cat((start_tok, labels), dim=2)[:, :, :-1, :]

        # LSTM forward pass
        lengths = label_mask.sum(dim=-1)
        outputs = []
        labels = self.input_dropout(labels)
        for i in range(N):
            # *ignore empty captions
            dense_mask = trans_mask[:, i]
            dense_labels = mask_select(labels[:, i, :, :], dense_mask)
            dense_lengths = mask_select(lengths[:, i], dense_mask)
            if dense_lengths.sum().item() > 0:
                input_unit = pack_padded_sequence(
                    dense_labels,
                    dense_lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                output, _ = self.lstms[i](input_unit, (h0, c0))
                output, _ = pad_packed_sequence(
                    output, batch_first=True, total_length=L
                )
                # *fill back empty lines with 0
                output = mask_unselect(output, dense_mask)
            else:
                output = torch.zeros((B, L, self.hidden_dim)).to(features)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.output_dropout(outputs)

        # map into words
        outputs = self.linear(outputs)

        return outputs

    def inference(
        lstm_state: torch.Tensor,
        features: torch.Tensor,
        trans_mask: torch.Tensor,
    ):
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

        self.embed = nn.Embedding(N_WORDS, hidden_dim)
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
        trans_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        if label_ids is None:
            return self.inference(context, trans_mask)

        B, N, L = label_ids.size()
        assert label_mask.size() == label_ids.size()

        # prepare inputs
        labels = self.embed(label_ids)
        labels = self.dropout_embed(labels)
        context = repeat(
            context, "B N d -> B N L d", B=B, N=N, L=L, d=self.context_dim
        )
        labels = torch.cat([context, labels], dim=-1)

        # lstm forward pass
        labels = rearrange(
            labels, "B N L d -> (B N) L d", B=B, N=N, L=L, d=self.lstm_input_dim
        )
        lengths = rearrange(label_mask.sum(dim=-1), "B N -> (B N)", B=B, N=N)
        # *ignore empty lines
        trans_mask = rearrange(trans_mask, "B N -> (B N)", B=B, N=N)
        labels = mask_select(labels, trans_mask)
        lengths = mask_select(lengths, trans_mask)
        labels = pack_padded_sequence(
            labels, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(labels)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=L
        )

        # map into words
        output = self.dropout_lstm(output)
        output = self.linear(output)
        # *fill back empty lines with zeros
        output = mask_unselect(output, trans_mask)
        output = rearrange(
            output, "(B N) L d -> B N L d", B=B, N=N, L=L, d=N_WORDS
        )

        return output

    def inference(context: torch.Tensor, trans_mask: torch.Tensor):
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

        self.embed = nn.Embedding(N_WORDS, hidden_dim)
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
        trans_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        if label_ids is None:
            return self.inference(context, trans_mask)

        B, N, L = label_ids.size()

        # prepare inputs
        start_tok = self.context_project(context)
        start_tok = start_tok.unsqueeze(2)
        labels = self.embed(label_ids)
        labels = torch.cat((start_tok, labels), dim=2)[:, :, :-1, :]
        labels = rearrange(
            labels, "B N L d -> (B N) L d", B=B, N=N, L=L, d=self.hidden_dim
        )
        label_mask = rearrange(label_mask, "B N L -> (B N) L", B=B, N=N, L=L)
        # *ignore empty lines
        trans_mask = rearrange(trans_mask, "B N -> (B N)", B=B, N=N)
        labels = mask_select(labels, trans_mask)
        label_mask = mask_select(label_mask, trans_mask)

        # positional embedding
        labels = labels + self.pos_embed(labels)
        labels = self.dropout_embed(labels)

        # decoder forward pass
        output = self.decoder(labels, mask=label_mask)
        output = self.norm(output)

        # map into words
        logits = self.to_logits(output)
        # *fill back empty lines with zeros
        logits = mask_unselect(logits, trans_mask)
        logits = rearrange(
            logits, "(B N) L d -> B N L d", B=B, N=N, L=L, d=N_WORDS
        )

        return logits

    def inference(self, context: torch.Tensor, trans_mask: torch.Tensor):
        pass

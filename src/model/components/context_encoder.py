"""Context encoders.

Input:
    image features: tensor of shape (B, N, d)

Output:
    context: tensor of shape (B, N, d)

Available context encoders:
    - simple_lstm
    - glocal
    - transformer
"""
import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    Encoder,
    FixedPositionalEmbedding,
    always,
)

logger = logging.getLogger(__name__)


class SimpleLSTMContext(nn.Module):
    """Contextualize, Show, and Tell.

    The winner of the VIST challenge.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        bidirectional=False,
        dropout=0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # try to act like tf.contrib.rnn.DropoutWrapper
        # https://github.com/dianaglzrico/neural-visual-storyteller/blob/master/show_and_tell_model.py#L370
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)

    @property
    def output_dim(self):
        return self.hidden_dim

    def forward(
        self, features: torch.Tensor, states_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, N, d)
        mask : torch.Tensor
            Tensor of shape (B, N)

        Returns
        -------
        Dict[str, torch.Tensor]
            context: torch.Tensor of shape (B, N, d)
            state: torch.Tensor of LSTM final state
        """
        _, N, _ = features.size()

        features = self.input_dropout(features)

        x_len = states_mask.sum(
            dim=1
        ).cpu()  # why .cpu(): https://github.com/pytorch/pytorch/issues/43227
        x_pack = pack_padded_sequence(
            features, x_len, batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.lstm(x_pack)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=N
        )

        output = self.output_dropout(output)

        return {
            "context": output,
            "state": (h_n, c_n),
        }


class GLocalContext(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.n_direction = 2 if bidirectional else 1
        self.linear_input_dim = hidden_dim * self.n_direction + input_dim
        self.linear_1 = nn.Sequential(
            nn.Linear(
                self.linear_input_dim,
                hidden_dim * self.n_direction,
            ),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(self.hidden_dim * self.n_direction, momentum=0.01),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(hidden_dim * self.n_direction, self.hidden_dim),
            nn.Dropout(p=dropout),
        )

    @property
    def output_dim(self):
        return self.hidden_dim

    def forward(
        self, features: torch.Tensor, states_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, N, d)
        mask : torch.Tensor
            Tensor of shape (B, N)

        Returns
        -------
        Dict[str, torch.Tensor]
            context: torch.Tensor of shape (B, N, d)
            state: torch.Tensor of LSTM final state
        """
        B, N, _ = features.size()

        x_len = states_mask.sum(dim=1).cpu()
        x_pack = pack_padded_sequence(
            features, x_len, batch_first=True, enforce_sorted=False
        )
        global_rnn, (h_n, c_n) = self.lstm(x_pack)
        global_rnn, _ = pad_packed_sequence(
            global_rnn, batch_first=True, total_length=N
        )
        glocal = torch.cat((features, global_rnn), dim=2).contiguous()

        glocal = rearrange(
            glocal, "B N d -> (B N) d", B=B, N=N, d=self.linear_input_dim
        )
        output = self.linear_1(glocal)
        output = self.linear_2(output)
        output = rearrange(
            output, "(B N) d -> B N d", B=B, N=N, d=self.output_dim
        )
        return {"context": output, "state": (h_n, c_n)}


class TransformerContext(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        position_embedding: str = "relative",
        max_seq_len: int = 5,
        encoder_args: Dict[str, Any] = {},
    ):
        """Transformer context encoder.

        Parameters
        ----------
        input_dim : int, optional
            dimension of input features, by default 512
        num_layers : int, optional
            number of layers, by default 2
        heads : int, optional
            number of attention heads, by default 8
        dropout : float, optional
            dropout probability, by default 0.1
        position_encoding : str, optional
            type of positional encoding, by default "relative", available
            options: "fixed", "absolute", "infused_fixed", "relative"
        max_seq_len : int, optional
            maximum sequence length, required when using absolute positional
            encoding, by default 5
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.position_embedding = position_embedding
        self.max_seq_len = max_seq_len

        self.pos_embed = self.setup_position_embedding()
        self.encoder = Encoder(
            dim=input_dim,
            depth=num_layers,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rel_pos_bias=self.rel_pos_bias,
            position_infused_attn=self.position_infused,
            **encoder_args
        )
        self.norm = nn.LayerNorm(input_dim)

    @property
    def output_dim(self):
        return self.input_dim

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
            pos_embed = FixedPositionalEmbedding(self.input_dim)
        elif self.position_embedding == "absolute":
            pos_embed = AbsolutePositionalEmbedding(
                self.input_dim, self.max_seq_len
            )
        else:
            pos_embed = always(0)
        return pos_embed

    def forward(
        self, features: torch.Tensor, states_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, N, d)
        mask : torch.Tensor
            Tensor of shape (B, N)

        Returns
        -------
        Dict[str, torch.Tensor]
            context: torch.Tensor of shape (B, N, d)
            state: torch.Tensor of LSTM final state
        """
        features = features + self.pos_embed(features)
        features, intermediates = self.encoder(
            features, mask=states_mask, return_hiddens=True
        )
        features = self.norm(features)
        attn_maps = list(
            map(
                lambda t: t.post_softmax_attn,
                intermediates.attn_intermediates,
            )
        )
        return {"context": features, "attention": attn_maps}


class FuseDiffContext(TransformerContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_diff = nn.Linear(self.input_dim * 2, self.input_dim)

    def forward(
        self,
        features: torch.Tensor,
        diff_features: torch.Tensor,
        states_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        features = torch.cat((features, diff_features), dim=-1)
        features = self.project_diff(features)
        return super().forward(features, states_mask)


class AttentionDiffContext(TransformerContext):
    def __init__(self, *args, diff_first=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.diff_first = diff_first
        self.segment_embed = nn.Embedding(2, self.input_dim)

    def forward(
        self,
        features: torch.Tensor,
        diff_features: torch.Tensor,
        states_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        B, N, _ = features.size()

        features = (
            features
            + self.pos_embed(features)
            + self.segment_embed(
                torch.zeros((B, N), device=features.device, dtype=torch.long)
            )
        )
        diff_features = (
            diff_features
            + self.pos_embed(diff_features)
            + self.segment_embed(
                torch.ones((B, N), device=features.device, dtype=torch.long)
            )
        )
        if self.diff_first:
            concat_features = torch.cat((diff_features, features), dim=1)
        else:
            concat_features = torch.cat((features, diff_features), dim=1)
        concat_mask = torch.cat((states_mask, states_mask), dim=1)
        features, intermediates = self.encoder(
            concat_features,
            mask=concat_mask,
            return_hiddens=True,
        )
        features = self.norm(features)
        attn_maps = list(
            map(
                lambda t: t.post_softmax_attn,
                intermediates.attn_intermediates,
            )
        )
        return {"context": features, "attention": attn_maps}


class CrossDiffContext(TransformerContext):
    def __init__(self, *args, diff_first=True, **kwargs):
        if "encoder_args" not in kwargs:
            kwargs["encoder_args"] = {}
        kwargs["encoder_args"]["cross_attend"] = True
        super().__init__(*args, **kwargs)
        self.diff_first = diff_first

    def forward(
        self,
        features: torch.Tensor,
        diff_features: torch.Tensor,
        states_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        features = features + self.pos_embed(features)
        diff_features = diff_features + self.pos_embed(diff_features)
        if self.diff_first:
            features, intermediates = self.encoder(
                diff_features,
                context=features,
                mask=states_mask,
                context_mask=states_mask,
                return_hiddens=True,
            )
        else:
            features, intermediates = self.encoder(
                features,
                context=diff_features,
                mask=states_mask,
                context_mask=states_mask,
                return_hiddens=True,
            )
        features = self.norm(features)
        attn_maps = list(
            map(
                lambda t: t.post_softmax_attn,
                intermediates.attn_intermediates,
            )
        )
        return {"context": features, "attention": attn_maps}


class AttentionBiDiffContext(TransformerContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.segment_embed = nn.Embedding(3, self.input_dim)

    def forward(
        self,
        features: torch.Tensor,
        early_diff_features: torch.Tensor,
        late_diff_features: torch.Tensor,
        states_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        B, N, _ = features.size()

        features = (
            features
            + self.pos_embed(features)
            + self.segment_embed(
                torch.empty(
                    (B, N), device=features.device, dtype=torch.long
                ).fill_(0)
            )
        )
        early_diff_features = (
            early_diff_features
            + self.pos_embed(early_diff_features)
            + self.segment_embed(
                torch.empty(
                    (B, N), device=features.device, dtype=torch.long
                ).fill_(1)
            )
        )
        late_diff_features = (
            late_diff_features
            + self.pos_embed(late_diff_features)
            + self.segment_embed(
                torch.empty(
                    (B, N), device=features.device, dtype=torch.long
                ).fill_(2)
            )
        )

        concat_features = torch.cat(
            (features, early_diff_features, late_diff_features), dim=1
        )
        concat_mask = torch.cat((states_mask, states_mask, states_mask), dim=1)
        features, intermediates = self.encoder(
            concat_features,
            mask=concat_mask,
            return_hiddens=True,
        )
        features = self.norm(features)
        attn_maps = list(
            map(
                lambda t: t.post_softmax_attn,
                intermediates.attn_intermediates,
            )
        )
        return {"context": features, "attention": attn_maps}


class PastFutureContext(nn.Module):

    EPSILON = 1e-8

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        normalize_weight: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.normalize_weight = normalize_weight

        self.project_weight = nn.Linear(self.input_dim, self.hidden_dim)
        self.project_feature = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        states_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        B, N, d = features.size()

        # B N N
        weight = torch.einsum(
            "bid,bjd->bij", self.project_weight(features), features
        )

        # B N N
        mask = torch.matmul(
            states_mask.unsqueeze(2).float(), states_mask.unsqueeze(1).float()
        )
        past_mask = (
            repeat(
                torch.tril(
                    torch.ones(N, N, device=features.device), diagonal=-1
                ).float(),
                "i j -> b i j",
                b=B,
                i=N,
                j=N,
            )
            * mask
        )
        future_mask = (
            repeat(
                torch.triu(
                    torch.ones(N, N, device=features.device), diagonal=1
                ).float(),
                "i j -> b i j",
                b=B,
                i=N,
                j=N,
            )
            * mask
        )
        past_weight = weight * past_mask
        past_weight = past_weight / (
            past_mask.sum(dim=2, keepdim=True) + self.EPSILON
        )
        future_weight = weight * future_mask
        future_weight = future_weight / (
            future_mask.sum(dim=2, keepdim=True) + self.EPSILON
        )

        # B N N d
        repeat_features = repeat(
            features, "B N d -> B r N d", B=B, N=N, r=N, d=d
        )

        if self.normalize_weight:
            past_features = (
                repeat_features
                * (F.softmax(past_weight, dim=-1) * past_mask)[..., None]
            ).sum(dim=2)
            future_features = (
                repeat_features
                * (F.softmax(future_weight, dim=-1) * future_mask)[..., None]
            ).sum(dim=2)
        else:
            past_features = (repeat_features * past_weight[..., None]).sum(
                dim=2
            )
            future_features = (repeat_features * future_weight[..., None]).sum(
                dim=2
            )

        context = torch.cat((past_features, features, future_features), dim=-1)
        context = self.project_feature(context)
        return {"context": context}

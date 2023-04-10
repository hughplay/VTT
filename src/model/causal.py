"""Nodes Definition.

Observable Nodes
---
- images: observable, corresponding to states
- transformations: observable,
- topic: observable,

- states: latent, corresponding to
- pattern: latent,
"""
from typing import Any, Dict

import torch
from torch import nn

from src.utils.arraytool import mask_select, mask_unselect

from .components.clip_text_autoencoder import CLIPAutoEncoder
from .components.stable_diffusion_autoencoder import StableDiffusionAutoEncoder
from .components.x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    Encoder,
    FixedPositionalEmbedding,
    always,
)

# Encoding modules


class Z_S1_T0_to_S0(nn.Module):
    def __init__(self, c_noise=512, c_state=512, c_trans=512):
        super().__init__()
        self.c_noise = c_noise
        self.c_state = c_state
        self.c_trans = c_trans
        self.fn = nn.Linear(self.c_noise + self.c_state, self.c_trans)

    def forward(self, z, s1, t0):
        return self.fn(torch.cat([z, s1, t0], dim=-1))


class Spre_Snext_Tpre_Tnext_to_S(nn.Module):
    def __init__(self, c_state=512, c_trans=512):
        super().__init__()
        self.c_state = c_state
        self.c_trans = c_trans

        self.empty_state = nn.Parameter(torch.zeros(1, self.c_state))
        self.empty_trans = nn.Parameter(torch.zeros(1, self.c_trans))

        # initialize empty state and transformation
        nn.init.normal_(self.empty_state)
        nn.init.normal_(self.empty_trans)

        self.fn = nn.Linear(self.c_state * 2 + self.c_trans * 2, self.c_state)

    def forward(self, s_pre, t_pre, s_next, t_next):
        return self.fn(torch.cat([s_pre, t_pre, s_next, t_next], dim=-1))


class Spre_Snext_Z_to_T(nn.Module):
    def __init__(self, c_noise=512, c_state=512, c_trans=512):
        super().__init__()
        self.c_noise = c_noise
        self.c_state = c_state
        self.c_trans = c_trans
        self.fn = nn.Linear(self.c_noise + self.c_state * 2, self.c_trans)

    def forward(self, z, s_pre, s_next):
        return self.fn(torch.cat([z, s_pre, s_next], dim=-1))


class S0_Ts_to_Z(nn.Module):
    def __init__(
        self,
        c_noise: int = 512,
        c_state: int = 512,
        c_trans: int = 512,
        num_layers: int = 1,
        heads: int = 8,
        dropout: float = 0.1,
        position_embedding: str = "relative",
        max_trans: int = 12,
        encoder_args: Dict[str, Any] = {},
    ):
        super().__init__()
        self.c_noise = c_noise
        self.c_state = c_state
        self.c_trans = c_trans

        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.position_embedding = position_embedding
        self.max_trans = max_trans
        self.max_seq_len = self.max_trans + 1

        self.en_s0 = nn.Linear(self.c_state, self.c_noise)
        self.en_trans = (
            nn.Linear(self.c_trans, self.c_noise)
            if self.c_trans != self.c_noise
            else nn.Identity()
        )

        self.pos_embed = self.setup_position_embedding()
        self.encoder = Encoder(
            dim=self.c_noise,
            depth=num_layers,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rel_pos_bias=self.rel_pos_bias,
            position_infused_attn=self.position_infused,
            **encoder_args,
        )

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
        self,
        s0: torch.Tensor,
        trans_list: torch.Tensor,
        trans_mask: torch.Tensor,
    ):
        s0 = self.en_s0(s0).unsqueeze(1)
        trans = self.en_trans(trans_list)
        trans = mask_select(trans, trans_mask)
        trans = self.en_trans(trans)
        trans = mask_unselect(trans, trans_mask)

        x = torch.cat([s0, trans], dim=1)
        x = x + self.pos_embed(x)

        x = self.encoder(x, mask=trans_mask)

        z = x[:, 0, ...]

        return z


# Decoding modules


class Z_to_S0(nn.Module):
    def __init__(self, c_noise=512, c_state=512):
        super().__init__()
        self.c_noise = c_noise
        self.c_state = c_state
        self.fn = nn.Linear(self.c_noise, self.c_state)

    def forward(self, z):
        return self.fn(z)


class S_T_to_S(nn.Module):
    def __init__(self, c_state=512, c_trans=512):
        super().__init__()
        self.c_state = c_state
        self.c_trans = c_trans
        self.fn = nn.Linear(self.c_state + self.c_trans, self.c_state)

    def forward(self, s, t):
        return self.fn(torch.cat([s, t], dim=-1))


class Z_S_to_T(nn.Module):
    def __init__(self, c_noise=512, c_state=512, c_trans=512):
        super().__init__()
        self.c_noise = c_noise
        self.c_state = c_state
        self.c_trans = c_trans
        self.fn = nn.Linear(self.c_noise + self.c_state, self.c_trans)

    def forward(self, z, s):
        return self.fn(torch.cat([z, s], dim=-1))


# class TransformaitonAutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()


# class TopicEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fn_topic = None

#     def forward(self, initial_state, transformations):
#         topic = self.fn_topic(initial_state, transformations)
#         return topic


# class StateDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fn_initial_state = None
#         self.fn_state = None

#     def forward(self, transformations, topic):
#         initial_state = self.fn_initial_state(topic)
#         states = [initial_state]
#         for transformation in transformations:
#             state = self.fn_state(states[-1], transformation)
#             states.append(state)
#         return states


# class TransformationDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fn_transformation = None

#     def forward(self, states, topic):
#         transformations = []
#         for state in states:
#             transformation = self.fn_transformation(state, topic)
#             transformations.append(transformation)
#         return transformations


class TransformationAutoEncoder(nn.Module):
    def __init__(self, c_noise=512):
        super().__init__()

        self.image_ae = StableDiffusionAutoEncoder()
        self.text_ae = CLIPAutoEncoder()

        self.c_state = self.image_ae.c_embed
        self.c_trans = self.text_ae.c_embed
        self.c_noise = c_noise

        # s: state, t: transformation, z: noise

        # encoding
        self.z_t_s1_to_s0 = Z_S1_T0_to_S0(
            c_noise=self.c_noise, c_state=self.c_state, c_trans=self.c_trans
        )
        self.spre_snext_tpre_tnext_to_s = Spre_Snext_Tpre_Tnext_to_S(
            c_state=self.c_state, c_trans=self.c_trans
        )

        # decoding
        self.z_to_s0 = Z_to_S0(c_noise=self.c_noise, c_state=self.c_state)
        self.s_t_to_s = S_T_to_S(c_state=self.c_state, c_trans=self.c_trans)
        self.z_s_to_t = Z_S_to_T(
            c_noise=self.c_noise, c_state=self.c_state, c_trans=self.c_trans
        )

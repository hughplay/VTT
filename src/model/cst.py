import torch
from torch import nn

from .components.context_encoder import SimpleLSTMContext
from .components.image_encoder import ImageEncoder
from .components.text_decoder import IndependentLSTMText


class CST(nn.Module):
    def __init__(
        self,
        dim=512,
        finetune_image_encoder=False,
        max_transformations=12,
        num_lstm_layers=1,
        lstm_dropout=0.3,
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name="inception_v3",
            finetune=finetune_image_encoder,
            output_dim=dim,
            batch_norm=False,
        )
        self.context_encoder = SimpleLSTMContext(
            input_dim=dim,
            hidden_dim=dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
        )
        self.decoder = IndependentLSTMText(
            state_dim=dim,
            feature_dim=dim,
            max_seq_len=max_transformations,
            word_emb_dim=dim,
            hidden_dim=dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
        )

    def forward(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        features = self.image_encoder(states)
        context = self.context_encoder(features, states_mask)
        end_states = features[:, 1:, :]
        logits = self.decoder(
            context["state"],
            end_states,
            states_mask[:, 1:],
            label_ids,
            label_mask,
        )
        return {
            "features": features,
            "context": context["context"],
            "logits": logits,
        }

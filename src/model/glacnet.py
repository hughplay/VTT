import torch
from torch import nn

from .components.context_encoder import GLocalContext
from .components.image_encoder import ImageEncoder
from .components.text_decoder import ContextLSTMText


class GLACNet(nn.Module):
    def __init__(
        self,
        dim=512,
        finetune_image_encoder=False,
        num_lstm_layers=2,
        embed_dropout=0.1,
        lstm_dropout=0.5,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name="resnet152",
            finetune=finetune_image_encoder,
            output_dim=dim,
            batch_norm=True,
        )
        self.context_encoder = GLocalContext(
            input_dim=dim,
            hidden_dim=dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.decoder = ContextLSTMText(
            context_dim=dim,
            word_emb_dim=dim,
            hidden_dim=dim,
            num_layers=num_lstm_layers,
            embed_dropout=embed_dropout,
            lstm_dropout=lstm_dropout,
            generate_cfg=generate_cfg,
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
        end_context = context["context"][:, 1:, :]
        outputs = self.decoder(
            end_context,
            states_mask[:, 1:],
            label_ids,
            label_mask,
            return_dict=True,
        )
        return {"features": features, "context": context["context"], **outputs}

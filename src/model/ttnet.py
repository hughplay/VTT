from torch import nn

from .components.context_encoder import TransformerContext
from .components.image_encoder import ImageEncoder
from .components.text_decoder import TransformerText


class TTNet(nn.Module):
    def __init__(
        self,
        dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name="ViT-B/32",
            finetune=finetune_image_encoder,
            output_dim=dim,
        )
        self.context_encoder = TransformerContext(
            input_dim=dim,
            num_layers=num_context_layers,
            position_embedding=context_pos_emb,
            max_seq_len=max_transformations + 1,
        )
        self.decoder = TransformerText(
            context_dim=dim,
            hidden_dim=dim,
            num_layers=num_decoder_layers,
            position_embedding=decoder_pos_emb,
            max_words=max_words,
        )

    def forward(self, states, states_mask, captions, captions_mask):
        features = self.image_encoder(states)
        context = self.context_encoder(features, states_mask)
        end_context = context["context"][:, 1:, :]
        logits = self.decoder(end_context, captions, captions_mask)
        return {
            "features": features,
            "context": context["context"],
            "logits": logits,
        }

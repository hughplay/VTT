import torch
from torch import nn

from .components.context_encoder import GLocalContext, TransformerContext
from .components.image_encoder import ImageEncoder
from .components.text_decoder import (
    ContextLSTMText,
    ContextTransformerText,
    TransformerText,
)


class TTNet(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
        tie_embedding=False,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name=image_encoder,
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
            tie_embedding=tie_embedding,
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


class TTNetGLocal(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        finetune_image_encoder=False,
        num_lstm_layers=2,
        lstm_dropout=0.5,
        num_decoder_layers=2,
        decoder_pos_emb="relative",
        max_words=24,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name=image_encoder,
            finetune=finetune_image_encoder,
            output_dim=dim,
        )
        self.context_encoder = GLocalContext(
            input_dim=dim,
            hidden_dim=dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.decoder = TransformerText(
            context_dim=dim,
            hidden_dim=dim,
            num_layers=num_decoder_layers,
            position_embedding=decoder_pos_emb,
            max_words=max_words,
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


class TTNetLSTM(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        context_pos_emb="relative",
        max_transformations=12,
        num_lstm_layers=2,
        embed_dropout=0.1,
        lstm_dropout=0.5,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name=image_encoder,
            finetune=finetune_image_encoder,
            output_dim=dim,
        )
        self.context_encoder = TransformerContext(
            input_dim=dim,
            num_layers=num_context_layers,
            position_embedding=context_pos_emb,
            max_seq_len=max_transformations + 1,
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


class TTNetContext(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        decoder_context_fusion="add",
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
        tie_embedding=False,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            name=image_encoder,
            finetune=finetune_image_encoder,
            output_dim=dim,
        )
        self.context_encoder = TransformerContext(
            input_dim=dim,
            num_layers=num_context_layers,
            position_embedding=context_pos_emb,
            max_seq_len=max_transformations + 1,
        )
        self.decoder = ContextTransformerText(
            context_dim=dim,
            hidden_dim=dim,
            fusion_mode=decoder_context_fusion,
            num_layers=num_decoder_layers,
            position_embedding=decoder_pos_emb,
            max_words=max_words,
            tie_embedding=tie_embedding,
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

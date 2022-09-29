import torch
from einops import repeat
from torch import nn

from src.dataset.vtt import CATEGORIES, TOPICS

from .components.context_encoder import (
    AttentionBiDiffContext,
    AttentionDiffContext,
    CrossDiffContext,
    FuseDiffContext,
    GLocalContext,
    TransformerContext,
)
from .components.image_encoder import (
    BiDiffImageEncoder,
    EarlyDiffImageEncoder,
    ImageEncoder,
    LateDiffImageEncoder,
)
from .components.text_decoder import (
    BiContextTransformerText,
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


class TTNetBiContext(nn.Module):
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
        self.decoder = BiContextTransformerText(
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
        global_context = context["context"][:, 0, :]
        outputs = self.decoder(
            global_context,
            end_context,
            states_mask[:, 1:],
            label_ids,
            label_mask,
            return_dict=True,
        )
        return {"features": features, "context": context["context"], **outputs}


class TTNetMultiTask(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        reconstruction_dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        decoder_context_fusion="add",
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
        tie_embedding=False,
        category_head=True,
        topic_head=True,
        head_dropout=0.0,
        reconstruction_head=True,
        bicontext=False,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.category_head = category_head
        self.topic_head = topic_head
        self.reconstruction_head = reconstruction_head

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
        self.bicontext = bicontext
        TextDecoder = (
            BiContextTransformerText
            if self.bicontext
            else ContextTransformerText
        )
        self.decoder = TextDecoder(
            context_dim=dim,
            hidden_dim=dim,
            fusion_mode=decoder_context_fusion,
            num_layers=num_decoder_layers,
            position_embedding=decoder_pos_emb,
            max_words=max_words,
            tie_embedding=tie_embedding,
            generate_cfg=generate_cfg,
        )

        if self.category_head:
            self.project_category = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(CATEGORIES))
            )
        if self.topic_head:
            self.project_topic = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(TOPICS))
            )
        if self.reconstruction_head:
            self.project_reconstruction = (
                nn.Linear(dim, reconstruction_dim)
                if dim != reconstruction_dim
                else nn.Identity()
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

        global_context = context["context"][:, 0, :]
        end_context = context["context"][:, 1:, :]
        if self.bicontext:
            outputs = self.decoder(
                global_context,
                end_context,
                states_mask[:, 1:],
                label_ids,
                label_mask,
                return_dict=True,
            )
        else:
            outputs = self.decoder(
                end_context,
                states_mask[:, 1:],
                label_ids,
                label_mask,
                return_dict=True,
            )

        outputs = {
            "features": features,
            "context": context["context"],
            **outputs,
        }

        if self.category_head:
            category = self.project_category(global_context)
            outputs["category_logits"] = category
        if self.topic_head:
            topic = self.project_topic(global_context)
            outputs["topic_logits"] = topic

        if self.reconstruction_head:
            reconstruction = self.project_reconstruction(end_context)
            outputs["reconstruction"] = reconstruction

        return outputs


class TTNetMTM(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        dim=512,
        reconstruction_dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        decoder_context_fusion="add",
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
        tie_embedding=False,
        category_head=True,
        topic_head=True,
        head_dropout=0.0,
        reconstruction_head=True,
        learned_mask=False,
        mask_ratio=-1.0,
        sample_mask_prob=1.0,
        zero_prob=1.0,
        random_prob=0.0,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.category_head = category_head
        self.topic_head = topic_head
        self.reconstruction_head = reconstruction_head

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

        if self.category_head:
            self.project_category = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(CATEGORIES))
            )
        if self.topic_head:
            self.project_topic = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(TOPICS))
            )
        if self.reconstruction_head:
            self.project_reconstruction = (
                nn.Linear(dim, reconstruction_dim)
                if dim != reconstruction_dim
                else nn.Identity()
            )

        self.learned_mask = learned_mask
        self.mask_ratio = min(1.0, mask_ratio)
        self.sample_mask_prob = max(0.0, min(1.0, sample_mask_prob))
        self.zero_prob = max(0.0, min(1.0, zero_prob))
        self.random_prob = max(0.0, min(1.0 - self.zero_prob, random_prob))

        if self.learned_mask:
            self.mask = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.mask)

    def forward(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        features = self.image_encoder(states)

        if self.training and self.mask_ratio > 0:
            # sample wise mask
            mask_pos = (
                torch.rand(states_mask.size(0), 1, device=states_mask.device)
                < self.sample_mask_prob
            )
            prob = torch.rand(states_mask.size(), device=states_mask.device)
            prob = prob / self.mask_ratio
            # unchanged, [zero_prob + random_prob, 1)
            # zero, [random_prob, random_prob + zero_prob)
            pos = prob < (self.zero_prob + self.random_prob)
            if self.learned_mask:
                n_mask, d = features[pos & mask_pos].size()
                mask = repeat(self.mask, "d -> n d", n=n_mask, d=d)
                features[pos & mask_pos] = mask
            else:
                features[pos & mask_pos] = 0
            # random, [0, random_prob)
            pos = prob < self.random_prob
            features[pos & mask_pos] = torch.rand_like(features[pos & mask_pos])

        context = self.context_encoder(features, states_mask)

        global_context = context["context"][:, 0, :]
        end_context = context["context"][:, 1:, :]
        outputs = self.decoder(
            end_context,
            states_mask[:, 1:],
            label_ids,
            label_mask,
            return_dict=True,
        )

        outputs = {
            "features": features,
            "context": context["context"],
            **outputs,
        }

        if self.category_head:
            category = self.project_category(global_context)
            outputs["category_logits"] = category
        if self.topic_head:
            topic = self.project_topic(global_context)
            outputs["topic_logits"] = topic

        if self.reconstruction_head:
            reconstruction = self.project_reconstruction(end_context)
            outputs["reconstruction"] = reconstruction

        return outputs


class TTNetDiff(nn.Module):
    def __init__(
        self,
        image_encoder: str = "ViT-B/32",
        diff_mode: str = "early",
        diff_only: bool = False,
        context_mode: str = "fuse",
        diff_first: bool = False,
        dim=512,
        reconstruction_dim=512,
        finetune_image_encoder=False,
        num_context_layers=2,
        num_decoder_layers=2,
        decoder_context_fusion="add",
        context_pos_emb="relative",
        decoder_pos_emb="relative",
        max_transformations=12,
        max_words=24,
        tie_embedding=False,
        category_head=True,
        topic_head=True,
        head_dropout=0.0,
        reconstruction_head=True,
        learned_mask=False,
        mask_ratio=-1.0,
        sample_mask_prob=1.0,
        zero_prob=1.0,
        random_prob=0.0,
        generate_cfg={},
    ) -> None:
        super().__init__()

        self.category_head = category_head
        self.topic_head = topic_head
        self.reconstruction_head = reconstruction_head

        if diff_mode == "early":
            DiffEncoder = EarlyDiffImageEncoder
        elif diff_mode == "late":
            DiffEncoder = LateDiffImageEncoder
        elif diff_mode == "early_and_late":
            DiffEncoder = BiDiffImageEncoder
        else:
            raise ValueError(f"Unknown diff mode {diff_mode}")
        self.diff_only = diff_only
        self.image_encoder = DiffEncoder(
            name=image_encoder,
            finetune=finetune_image_encoder,
            output_dim=dim,
        )

        if context_mode == "fuse":
            self.context_encoder = FuseDiffContext(
                input_dim=dim,
                num_layers=num_context_layers,
                position_embedding=context_pos_emb,
                max_seq_len=max_transformations + 1,
            )
        elif context_mode == "attention":
            if diff_only:
                self.context_encoder = TransformerContext(
                    input_dim=dim,
                    num_layers=num_context_layers,
                    position_embedding=context_pos_emb,
                    max_seq_len=max_transformations + 1,
                )
            elif diff_mode == "early_and_late":
                self.context_encoder = AttentionBiDiffContext(
                    input_dim=dim,
                    num_layers=num_context_layers,
                    position_embedding=context_pos_emb,
                    max_seq_len=max_transformations + 1,
                )
            else:
                self.context_encoder = AttentionDiffContext(
                    input_dim=dim,
                    num_layers=num_context_layers,
                    position_embedding=context_pos_emb,
                    max_seq_len=max_transformations + 1,
                    diff_first=diff_first,
                )
        elif context_mode == "cross":
            self.context_encoder = CrossDiffContext(
                input_dim=dim,
                num_layers=num_context_layers,
                position_embedding=context_pos_emb,
                max_seq_len=max_transformations + 1,
                diff_first=diff_first,
            )
        else:
            raise ValueError(f"Unknown context mode {context_mode}")

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

        if self.category_head:
            self.project_category = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(CATEGORIES))
            )
        if self.topic_head:
            self.project_topic = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(dim, len(TOPICS))
            )
        if self.reconstruction_head:
            self.project_reconstruction = (
                nn.Linear(dim, reconstruction_dim)
                if dim != reconstruction_dim
                else nn.Identity()
            )

        self.learned_mask = learned_mask
        self.mask_ratio = min(1.0, mask_ratio)
        self.sample_mask_prob = max(0.0, min(1.0, sample_mask_prob))
        self.zero_prob = max(0.0, min(1.0, zero_prob))
        self.random_prob = max(0.0, min(1.0 - self.zero_prob, random_prob))

        if self.learned_mask:
            self.mask = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.mask)

    def forward(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        label_ids: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ):
        features_list = self.image_encoder(states, states_mask)
        if self.diff_only:
            features_list = features_list[1:]

        if self.training and self.mask_ratio > 0:
            # sample wise mask
            mask_pos = (
                torch.rand(states_mask.size(0), 1, device=states_mask.device)
                < self.sample_mask_prob
            )
            for features in features_list:
                prob = torch.rand(states_mask.size(), device=states_mask.device)
                prob = prob / self.mask_ratio
                # unchanged, [zero_prob + random_prob, 1)
                # zero, [random_prob, random_prob + zero_prob)
                pos = prob < (self.zero_prob + self.random_prob)
                if self.learned_mask:
                    n_mask, d = features[pos & mask_pos].size()
                    mask = repeat(self.mask, "d -> n d", n=n_mask, d=d)
                    features[pos & mask_pos] = mask
                else:
                    features[pos & mask_pos] = 0
                # random, [0, random_prob)
                pos = prob < self.random_prob
                features[pos & mask_pos] = torch.rand_like(
                    features[pos & mask_pos]
                )

        context = self.context_encoder(*features_list, states_mask)

        global_context = context["context"][:, 0, :]
        end_context = context["context"][:, 1 : states.size(1), :]
        outputs = self.decoder(
            end_context,
            states_mask[:, 1:],
            label_ids,
            label_mask,
            return_dict=True,
        )

        outputs = {
            "features_list": features_list,
            "context": context["context"],
            **outputs,
        }

        if self.category_head:
            category = self.project_category(global_context)
            outputs["category_logits"] = category
        if self.topic_head:
            topic = self.project_topic(global_context)
            outputs["topic_logits"] = topic

        if self.reconstruction_head:
            reconstruction = self.project_reconstruction(end_context)
            outputs["reconstruction"] = reconstruction

        return outputs

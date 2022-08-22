from typing import Dict

import torch
from einops import rearrange
from torch import nn

from src.dataset.vtt import CATEGORIES, TOPICS
from src.model.components.text_encoder import TextCLIP

IGNORE_INDEX = -100


class GenerationLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        logits = outputs["logits"]
        logits = rearrange(logits, "B N L C -> B C N L")
        target = outputs["label_ids"]
        mask = outputs["label_mask"]

        target = target * mask + IGNORE_INDEX * (~mask)
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss}
        return loss


class CategoryLoss(nn.Module):
    def __init__(self, context_dim: int = 512, n_category=len(CATEGORIES)):
        super().__init__()

        self.linear = nn.Linear(context_dim, n_category)
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        global_feat = outputs["context"][:, 0, :]
        logits = self.linear(global_feat)
        target = outputs["category"]
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss, "logits": logits}
        return loss


class TopicLoss(nn.Module):
    def __init__(self, context_dim: int = 512, n_topic=len(TOPICS)):
        super().__init__()

        self.linear = nn.Linear(context_dim, n_topic)
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        # position #0 is the global feature
        global_feat = outputs["context"][:, 0, :]
        logits = self.linear(global_feat)
        target = outputs["topic"]
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss, "logits": logits}
        return loss


class ClassificationLoss(nn.Module):
    def __init__(
        self,
        context_dim: int = 512,
        n_category=len(CATEGORIES),
        n_topic=len(TOPICS),
        w_category=1.0,
        w_topic=1.0,
    ):
        super().__init__()

        self.w_category = w_category
        self.w_topic = w_topic
        self.category_loss = CategoryLoss(
            context_dim=context_dim, n_category=n_category
        )
        self.topic_loss = TopicLoss(context_dim=context_dim, n_topic=n_topic)

    def forward(
        self, outputs: Dict[str, torch.Tensor], return_dict: bool = False
    ) -> torch.Tensor:

        category_loss = self.category_loss(outputs, return_dict=True)
        topic_loss = self.topic_loss(outputs, return_dict=True)
        loss = (
            category_loss["loss"] * self.w_category
            + topic_loss["loss"] * self.w_topic
        ) / (self.w_category + self.w_topic)
        if return_dict:
            return {
                "category_logits": category_loss["logits"],
                "category_loss": category_loss["loss"],
                "topic_logits": topic_loss["logits"],
                "topic_loss": topic_loss["loss"],
                "loss": loss,
            }
        return loss


class TransformationConstructionLoss(nn.Module):
    def __init__(self, context_dim=512, name="ViT-L/14") -> None:
        super().__init__()

        self.loss = nn.MSELoss(reduction="none")
        self.clip = TextCLIP(name=name)
        self.context_project = (
            nn.Linear(context_dim, self.clip.output_dim)
            if context_dim != self.clip.output_dim
            else nn.Identity()
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        # position #0 is the global feature
        context = outputs["context"][:, 1:, :]
        target = outputs["label_ids"]
        mask = outputs["label_mask"].sum(dim=-1) > 0

        context = self.context_project(context)
        target = self.clip(target)
        loss = (
            self.loss(context, target).mean(dim=-1) * mask
        ).sum() / mask.sum()

        if return_dict:
            return {"loss": loss, "reconstruction": context}
        return loss


class TellingLossV1(nn.Module):
    def __init__(
        self,
        context_dim: int = 512,
        text_model: str = "ViT-L/14",
        w_generate: float = 1.0,
        w_classify: float = 1.0,
        w_construct: float = 1.0,
        w_category: float = 1.0,
        w_topic: float = 1.0,
    ):
        super().__init__()

        self.w_generate = w_generate
        self.w_classify = w_classify
        self.w_construct = w_construct

        self.generation_loss = GenerationLoss()
        self.classification_loss = ClassificationLoss(
            context_dim=context_dim, w_category=w_category, w_topic=w_topic
        )
        self.construction_loss = TransformationConstructionLoss(name=text_model)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:
        generation_loss = self.generation_loss(outputs, return_dict=True)
        classification_loss = self.classification_loss(
            outputs, return_dict=True
        )
        construction_loss = self.construction_loss(outputs, return_dict=True)
        loss = (
            generation_loss["loss"] * self.w_generate
            + classification_loss["loss"] * self.w_classify
            + construction_loss["loss"] * self.w_construct
        ) / (self.w_generate + self.w_classify + self.w_construct)
        if return_dict:
            res = {}
            res.update(generation_loss)
            res.update(classification_loss)
            res.update(construction_loss)
            res.update(
                {
                    "generation_loss": generation_loss["loss"],
                    "classification_loss": classification_loss["loss"],
                    "construction_loss": construction_loss["loss"],
                    "loss": loss,
                }
            )
            return res
        return loss

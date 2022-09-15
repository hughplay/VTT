from typing import Dict, Union

import torch
from einops import rearrange
from torch import nn

from src.model.components.text_encoder import TextCLIP

IGNORE_INDEX = -100


def shift_tensor(
    tensor: torch.Tensor,
    shift: int,
    dim: int,
    shift_fill: Union[float, int] = None,
) -> torch.Tensor:
    """Shift the tensor by a given amount."""
    if shift == 0:
        return tensor

    tensor = torch.roll(tensor, shift, dim)
    length = tensor.size(dim)

    if shift_fill is not None:
        if shift < 0:
            start, end = length + shift, length
        else:
            start, end = 0, shift
        tensor.index_fill_(
            dim, torch.arange(start, end, device=tensor.device), shift_fill
        )

    return tensor


class GenerationLoss(nn.Module):
    def __init__(self, logit_shift: int = 0, label_shift: int = -1):
        super().__init__()

        self.logit_shift = logit_shift
        self.label_shift = label_shift
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        logits = outputs["logits"]
        logits = shift_tensor(logits, self.logit_shift, dim=2)
        logits = rearrange(logits, "B N L C -> B C N L")

        target = outputs["label_ids"]
        target = shift_tensor(target, self.label_shift, dim=2)

        mask = outputs["label_mask"]
        mask = shift_tensor(mask, self.label_shift, dim=2, shift_fill=False)

        target = target * mask + IGNORE_INDEX * (~mask)
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss}
        return loss


class CategoryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        # global_feat = outputs["context"][:, 0, :]
        # logits = self.linear(global_feat)
        logits = outputs["category_logits"]
        target = outputs["category"]
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss}
        return loss


class TopicLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:

        # position #0 is the global feature
        # global_feat = outputs["context"][:, 0, :]
        # logits = self.linear(global_feat)
        logits = outputs["topic_logits"]
        target = outputs["topic"]
        loss = self.loss(logits, target)

        if return_dict:
            return {"loss": loss}
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, w_category=1.0, w_topic=1.0):
        super().__init__()

        self.loss_list = []

        self.w_category = w_category
        self.w_topic = w_topic

        if w_category is not None:
            self.category_loss = CategoryLoss()
            self.loss_list.append(("category", self.category_loss, w_category))
        if w_topic is not None:
            self.topic_loss = TopicLoss()
            self.loss_list.append(("topic", self.topic_loss, w_topic))

    def forward(
        self, outputs: Dict[str, torch.Tensor], return_dict: bool = False
    ) -> torch.Tensor:
        res = {}
        loss_total, w_total = 0.0, 0.0
        for name, loss_func, w_loss in self.loss_list:
            result = loss_func(outputs, return_dict=True)
            res.update(result)
            res[name + "_loss"] = result["loss"]
            loss_total += res[name + "_loss"] * w_loss
            w_total += w_loss
        res["loss"] = loss_total / w_total
        if return_dict:
            return res
        return res["loss"]


class TransformationConstructionLoss(nn.Module):
    def __init__(self, name="ViT-L/14") -> None:
        super().__init__()

        self.loss = nn.MSELoss(reduction="none")
        self.clip = TextCLIP(name=name)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:
        reconstruction = outputs["reconstruction"]
        assert reconstruction.size(-1) == self.clip.output_dim

        target = outputs["label_ids"]
        mask = outputs["label_mask"].sum(dim=-1) > 0

        target = self.clip(target)
        loss = (
            self.loss(reconstruction, target).mean(dim=-1) * mask
        ).sum() / mask.sum()

        if return_dict:
            return {"loss": loss}
        return loss


class TellingLossV1(nn.Module):
    def __init__(
        self,
        logit_shift: int = 0,
        label_shift: int = -1,
        text_model: str = "ViT-L/14",
        w_generate: float = 1.0,
        w_classify: float = 1.0,
        w_construct: float = 1.0,
        w_category: float = 1.0,
        w_topic: float = 1.0,
    ):
        super().__init__()

        self.loss_list = []

        if w_generate is not None:
            self.generation_loss = GenerationLoss(
                logit_shift=logit_shift, label_shift=label_shift
            )
            self.loss_list.append(
                ("generation", self.generation_loss, w_generate)
            )

        if w_classify is not None:
            self.classification_loss = ClassificationLoss(
                w_category=w_category,
                w_topic=w_topic,
            )
            self.loss_list.append(
                ("classification", self.classification_loss, w_classify)
            )

        if w_construct is not None:
            self.construction_loss = TransformationConstructionLoss(
                name=text_model
            )
            self.loss_list.append(
                ("construction", self.construction_loss, w_construct)
            )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> torch.Tensor:
        res = {}
        loss_total, w_total = 0.0, 0.0
        for name, loss_func, w_loss in self.loss_list:
            result = loss_func(outputs, return_dict=True)
            res.update(result)
            res[name + "_loss"] = result["loss"]
            loss_total += res[name + "_loss"] * w_loss
            w_total += w_loss
        res["loss"] = loss_total / w_total
        if return_dict:
            return res
        return res["loss"]

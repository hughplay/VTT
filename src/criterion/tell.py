import logging
from typing import Dict, List, Sequence, Union

import torch
from einops import rearrange
from torch import nn
from torchmetrics.utilities.data import dim_zero_cat

from src.dataset.text import SimpleTokenizer
from src.dataset.vtt import CATEGORIES, TOPICS
from src.utils.arraytool import tolist
from src.utils.datatool import write_jsonlines
from src.utils.timetool import with_time

from .components.metrics import (
    BLEU,
    METEOR,
    ROUGE,
    SPICE,
    Accuracy,
    BERTScore,
    CIDEr,
    Perplexity,
)

logger = logging.getLogger(__name__)


class TTCriterion(nn.Module):
    """Transformation Telling Criterion.

    The outputs should be a dictionary of torch.Tensor with the following keys:
    - "logits": the logits of the model
    - "label_ids": the targets captions
    - "context": the context, required by TellingLoss
    """

    def __init__(
        self,
        loss=None,
        bert_score_model="roberta-large",
        category: bool = False,
        topic: bool = False,
    ) -> None:
        super().__init__()

        self.loss = loss
        self.bert_score_model = bert_score_model
        self.category = category
        self.topic = topic

        self.tokenizer = SimpleTokenizer()

        # self.bleu_4 = BLEU(n=4)
        self.bleu_4 = BLEU(n_gram=4)
        self.rouge = ROUGE()
        self.meteor = METEOR()
        self.cider = CIDEr()
        self.spice = SPICE()
        self.bert_score = BERTScore(
            model_name_or_path=bert_score_model,
            rescale_with_baseline=True,
        )
        self.perplexity = Perplexity()

        self.train_metrics = [
            ("PPL", self.perplexity),
        ]
        self.eval_metrics = [
            ("BLEU_4", self.bleu_4),
            ("ROUGE", self.rouge),
            ("METEOR", self.meteor),
            ("CIDEr", self.cider),
            ("SPICE", self.spice),
            ("BERTScore", self.bert_score),
        ]

        self.classify_metrics = []
        if self.category:
            self.category_acc = Accuracy(
                num_classes=len(CATEGORIES), classes=CATEGORIES, multiclass=True
            )
            self.classify_metrics.append(("CategoryAcc", self.category_acc))
        if self.topic:
            self.topic_acc = Accuracy(
                num_classes=len(TOPICS), classes=TOPICS, multiclass=True
            )
            self.classify_metrics.append(("TopicAcc", self.topic_acc))

        self.samples = []

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        update_eval: bool = True,
        exclude_eval_metrics: Union[str, Sequence[str]] = None,
    ):
        if compute_loss:
            result = self.loss(outputs, return_dict=True)
            self.perplexity.update(result["loss"])
            outputs.update(result)

        if self.category and "category_logits" in outputs:
            category_pred = outputs["category_logits"].argmax(dim=-1)
            category_target = outputs["category"]
            self.category_acc.update(category_pred, category_target)
            category_pred = [CATEGORIES[x] for x in tolist(category_pred)]
            category_target = [CATEGORIES[x] for x in tolist(category_target)]
        else:
            category_pred, category_target = None, None

        if self.topic and "topic_logits" in outputs:
            topic_pred = outputs["topic_logits"].argmax(dim=-1)
            topic_target = outputs["topic"]
            self.topic_acc.update(topic_pred, topic_target)
            topic_pred = [TOPICS[x] for x in tolist(topic_pred)]
            topic_target = [TOPICS[x] for x in tolist(topic_target)]
        else:
            topic_pred, topic_target = None, None

        if update_eval:
            if "sequence" not in outputs:
                outputs["sequence"] = outputs["logits"].argmax(dim=-1)
            preds = rearrange(outputs["sequence"], "B N L -> (B N) L")
            target = rearrange(outputs["label_ids"], "B N L -> (B N) L")
            mask = rearrange(outputs["label_mask"].sum(-1) > 0, "B N -> (B N)")

            preds = [
                self.tokenizer.smart_decode(tolist(x)) for x in preds[mask]
            ]
            target = [
                self.tokenizer.smart_decode(tolist(x)) for x in target[mask]
            ]

            self.update(
                preds, target, exclude_eval_metrics=exclude_eval_metrics
            )
        else:
            preds, target = None, None

        self.record_samples(
            index=outputs["index"],
            trans_length=(outputs["label_mask"].sum(-1) > 0).sum(-1),
            trans_preds=preds,
            trans_target=target,
            category_pred=category_pred,
            category_target=category_target,
            topic_pred=topic_pred,
            topic_target=topic_target,
        )

        return outputs

    def record_samples(
        self,
        index: torch.Tensor,
        trans_length: torch.Tensor,
        trans_preds: List[str] = None,
        trans_target: List[str] = None,
        category_pred: List[str] = None,
        category_target: List[str] = None,
        topic_pred: List[str] = None,
        topic_target: List[str] = None,
    ) -> None:
        assert len(index) == len(trans_length)
        if trans_preds is not None and trans_target is not None:
            assert len(trans_preds) == len(trans_target) == trans_length.sum()
        if category_pred is not None and category_target is not None:
            assert len(category_pred) == len(category_target) == len(index)
        if topic_pred is not None and topic_target is not None:
            assert len(topic_pred) == len(topic_target) == len(index)

        curr_idx = 0
        for i, (idx, length) in enumerate(zip(index, trans_length)):

            sample = {"index": idx.item()}

            if trans_preds is not None and trans_target is not None:
                sample["preds"] = trans_preds[curr_idx : curr_idx + length]
                sample["label"] = trans_target[curr_idx : curr_idx + length]
            if category_pred is not None and category_target is not None:
                sample["category_pred"] = category_pred[i]
                sample["category_target"] = category_target[i]
            if topic_pred is not None and topic_target is not None:
                sample["topic_pred"] = topic_pred[i]
                sample["topic_target"] = topic_target[i]

            self.samples.append(sample)
            curr_idx += length

    def update(
        self,
        preds: Sequence[str],
        target: Sequence[str],
        exclude_eval_metrics: Union[str, Sequence[str]] = None,
    ):
        if exclude_eval_metrics is None:
            exclude_eval_metrics = []
        elif isinstance(exclude_eval_metrics, str):
            exclude_eval_metrics = [exclude_eval_metrics]

        eval_targets = [
            (p, f) for p, f in zip(preds, target) if len(p) > 0 or len(f) > 0
        ]
        preds, target = list(zip(*eval_targets))
        preds, target = list(preds), list(target)
        for metric_name, metric in self.eval_metrics:
            if metric_name in exclude_eval_metrics:
                continue
            metric.update(preds, target)

    def compute(self, verbose=False):
        metrics = {}
        for metric_name, metric in (
            self.train_metrics + self.eval_metrics + self.classify_metrics
        ):
            if metric._update_called:
                if verbose:
                    value, time_cost = with_time(
                        metric.compute, pretty_time=True
                    )()
                    logger.info(f"- {metric_name}: {value} ({time_cost})")
                else:
                    value = metric.compute()
                if value is not None:
                    metrics[metric_name] = value
        return metrics

    @property
    def scores(self):
        result = {}
        for metric_name, metric in self.train_metrics + self.eval_metrics:
            if metric._update_called and hasattr(metric, "scores"):
                if isinstance(metric.scores, list) and isinstance(
                    metric.scores[0], torch.Tensor
                ):
                    metric.scores = dim_zero_cat(metric.scores)
                result[metric_name] = tolist(metric.scores)
        return result

    def reset(self):
        for _, metric in (
            self.train_metrics + self.eval_metrics + self.classify_metrics
        ):
            metric.reset()
        self.samples = []

    def save(self, path: str = None):
        if path is None:
            path = "detail.jsonl"
        # tested only with one GPU
        total_sequence = sum(len(s["label"]) for s in self.samples)
        for metric_name, metric in self.eval_metrics:
            if metric._update_called and hasattr(metric, "scores"):
                if isinstance(metric.scores, list) and isinstance(
                    metric.scores[0], torch.Tensor
                ):
                    metric.scores = dim_zero_cat(metric.scores)
                assert len(metric.scores) == total_sequence
                curr_idx = 0
                for sample in self.samples:
                    n_trans = len(sample["label"])
                    sample[metric_name] = tolist(
                        metric.scores[curr_idx : curr_idx + n_trans]
                    )
                    curr_idx += n_trans

        write_jsonlines(path, self.samples)

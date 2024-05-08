import logging
from typing import Dict, List, Sequence, Union

import torch
from torch import nn
from torchmetrics.utilities.data import dim_zero_cat

from src.dataset.text import SimpleTokenizer
from src.utils.arraytool import tolist
from src.utils.datatool import write_jsonlines
from src.utils.timetool import with_time

from .components.metrics import (
    BLEU,
    METEOR,
    ROUGE,
    SPICE,
    BERTScore,
    CIDEr,
    Perplexity,
)

logger = logging.getLogger(__name__)


class TextCriterion(nn.Module):
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
    ) -> None:
        super().__init__()

        self.loss = loss

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

        if update_eval:
            if "sequence" not in outputs:
                outputs["sequence"] = outputs["logits"].argmax(dim=-1)
            preds = outputs["sequence"]
            target = outputs["label_ids"]

            preds = [self.tokenizer.smart_decode(tolist(x)) for x in preds]
            target = [self.tokenizer.smart_decode(tolist(x)) for x in target]

            self.update(
                preds, target, exclude_eval_metrics=exclude_eval_metrics
            )
            self.record_samples(
                index=outputs["index"],
                preds=preds,
                target=target,
            )
        else:
            preds, target = None, None

        return outputs

    def record_samples(
        self,
        index: torch.Tensor,
        preds: List[str] = None,
        target: List[str] = None,
    ) -> None:
        assert len(index) == len(preds)
        if preds is not None and target is not None:
            assert len(preds) == len(target)

        for idx, pred, tar in zip(tolist(index), preds, target):
            sample = {
                "index": idx,
                "preds": pred,
                "label": tar,
            }
            self.samples.append(sample)

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
        for metric_name, metric in self.train_metrics + self.eval_metrics:
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
        for _, metric in self.train_metrics + self.eval_metrics:
            metric.reset()
        self.samples = []

    def save(self, path: str = None):
        if path is None:
            path = "detail.jsonl"
        # tested only with one GPU
        total_sequence = len(self.samples)
        for metric_name, metric in self.eval_metrics:
            if metric._update_called and hasattr(metric, "scores"):
                if isinstance(metric.scores, list) and isinstance(
                    metric.scores[0], torch.Tensor
                ):
                    metric.scores = dim_zero_cat(metric.scores)
                metrics_scores = tolist(metric.scores)
                assert len(metrics_scores) == total_sequence
                for sample, score in zip(self.samples, metrics_scores):
                    sample[metric_name] = score

        write_jsonlines(path, self.samples)

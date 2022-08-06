import os
from typing import Dict, List, Sequence

import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.text.bert import BERTScore as TorchBERTScore
from torchmetrics.text.bleu import BLEUScore
from transformers import logging

from src.dataset.text import SimpleTokenizer

from .components.pycocoevalcap.cider.cider import CiderScorer as coco_cider
from .components.pycocoevalcap.meteor.meteor import Meteor as coco_meteor
from .components.pycocoevalcap.rouge.rouge import Rouge as coco_rouge
from .components.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# disable the warning from huggingface:
# The current process just got forked, after parallelism has already been used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# disable the warning from huggingface:
# Some weights of the model checkpoint at roberta-large were not used when
# initializing RobertaModel
logging.set_verbosity_error()


def coco_fmt_sequences(
    sequences: Sequence[str],
) -> Dict[str, List[Dict[str, str]]]:
    """Convert a list of sequences to the format required by the COCO
    metrics."""
    return {idx: [{"caption": s}] for idx, s in enumerate(sequences)}


def coco_extract_sequences(
    sequences: Dict[str, List[Dict[str, str]]]
) -> Sequence[str]:
    """Extract the sequences."""
    return [sequences[idx][0] for idx in range(len(sequences))]


class METEOR(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meteor = coco_meteor()
        self.tokenizer = PTBTokenizer()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:
        preds = self.tokenizer.tokenize(coco_fmt_sequences(preds))
        target = self.tokenizer.tokenize(coco_fmt_sequences(target))
        score, scores = self.meteor.compute_score(target, preds)
        self.score += score
        self.scores.extend(scores)
        self.score_count += len(scores)

    def compute(self):
        return self.score / self.score_count


class ROUGE(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rouge = coco_rouge()
        self.tokenizer = PTBTokenizer()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:
        preds = self.tokenizer.tokenize(coco_fmt_sequences(preds))
        target = self.tokenizer.tokenize(coco_fmt_sequences(target))
        score, scores = self.rouge.compute_score(target, preds)
        self.score += score
        self.scores.extend(scores)
        self.score_count += len(scores)

    def compute(self):
        return self.score / self.score_count


class CIDEr(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n=4, sigma=6.0, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.sigma = sigma
        self.tokenizer = PTBTokenizer()

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:
        preds = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(preds))
        )
        target = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(target))
        )
        self.preds.extend(preds)
        self.target.extend(target)

    def compute(self):
        cider = coco_cider(n=self.n, sigma=self.sigma)
        for p, t in zip(self.preds, self.target):
            cider += (p, [t])
        score, scores = cider.compute_score()
        self.scores.extend(scores)
        self.score = torch.tensor(score).to(self.score)
        return self.score


class BERTScore(TorchBERTScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def compute(self):
        result = super().compute()
        self.scores = result["f1"]
        self.score = torch.sum(torch.tensor(result["f1"]).to(self.score))
        self.score_count = torch.tensor(len(result["f1"])).to(self.score_count)
        return self.score / self.score_count


class TTCriterion(nn.Module):
    def __init__(self, loss=None) -> None:
        super().__init__()

        self.tokenizer = SimpleTokenizer()

        self.loss = loss

        self.bleu_4 = BLEUScore(n_gram=4)
        self.rouge = ROUGE()
        self.meteor = METEOR()
        self.cider = CIDEr()
        self.bert_score = BERTScore(model_name_or_path="roberta-large")

        self.metrics = [
            ("BLEU_4", self.bleu_4),
            ("ROUGE", self.rouge),
            ("METEOR", self.meteor),
            ("CIDEr", self.cider),
            ("BERTScore", self.bert_score),
        ]

    def forward(self, outputs, stage="train"):
        if stage == "train":
            loss = self.loss(outputs)
            outputs.update(loss=loss)
        else:
            preds, target = None
            self.update(preds, target)

    def update(self, preds, target):
        for metric_name, metric in self.metrics:
            if metric_name in ["BLEU_4"]:
                metric.update(preds, [target])
            else:
                metric.update(preds, target)

    def compute(self):
        metrics = {
            metric_name: metric.compute()
            for metric_name, metric in self.metrics
        }
        # metrics = {}
        # from src.utils.timetool import with_time
        # for metric_name, metric in self.metrics:
        #     res, seconds = with_time(metric.compute)()
        #     metrics[metric_name] = res
        #     print(f"{metric_name}: {seconds}")
        return metrics

    def reset(self):
        for _, metric in self.metrics:
            metric.reset()

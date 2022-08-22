import logging
import os
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torchmetrics import Metric
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.metric import jit_distributed_available
from torchmetrics.text.bert import BERTScore as TorchBERTScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.utilities.data import dim_zero_cat
from transformers import logging as transformers_logging

from src.dataset.text import SimpleTokenizer
from src.utils.timetool import with_time

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
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


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


def tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return [x]


def cat_states(func):
    def wrapper(self, *args, **kwargs):
        if not jit_distributed_available() and not self._is_synced:
            output_dict = {
                attr: getattr(self, attr) for attr in self._reductions
            }

            for attr, reduction_fn in self._reductions.items():
                # pre-concatenate metric states that are lists to reduce number of all_gather operations
                if (
                    reduction_fn == dim_zero_cat
                    and isinstance(output_dict[attr], list)
                    and len(output_dict[attr]) >= 1
                ):
                    setattr(self, attr, dim_zero_cat(output_dict[attr]))
        return func(self, *args, **kwargs)

    return wrapper


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
        self.scores.append(torch.tensor(scores).to(self.score.device))
        self.score_count += len(scores)

    @cat_states
    def compute(self):
        return self.score / self.score_count if self.score_count else None


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
        self.scores.append(torch.tensor(scores).to(self.score.device))
        self.score_count += len(scores)

    @cat_states
    def compute(self):
        return self.score / self.score_count if self.score_count else None


class CIDEr(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n=4, sigma=6.0, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.sigma = sigma
        self.tokenizer = PTBTokenizer()
        self.clip_tokenizer = SimpleTokenizer()

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds_offset", default=[], dist_reduce_fx="cat")
        self.add_state("target_offset", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.scores = []

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:

        preds = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(preds))
        )
        for pred in preds:
            pred = self.clip_tokenizer.encode(pred)
            self.preds_offset.append(
                torch.tensor(len(self.preds)).to(self.score.device)
            )
            self.preds.append(torch.tensor(pred).to(self.score.device))
        target = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(target))
        )
        for t in target:
            t = self.clip_tokenizer.encode(t)
            self.target_offset.append(
                torch.tensor(len(self.target)).to(self.score.device)
            )
            self.target.append(torch.tensor(t).to(self.score.device))

    @cat_states
    def compute(self):
        if len(self.preds) == 0:
            return None
        cider = coco_cider(n=self.n, sigma=self.sigma)
        start_p = 0
        start_t = 0
        for offset_p, offset_t in zip(self.preds_offset, self.target_offset):
            p = self.clip_tokenizer.decode(
                tolist(self.preds[start_p : start_p + offset_p])
            )
            t = self.clip_tokenizer.decode(
                tolist(self.target[start_t : start_t + offset_t])
            )
            cider += (p, [t])
            start_p += offset_p
            start_t += offset_t
        score, scores = cider.compute_score()
        self.scores = tolist(scores)
        self.score = torch.tensor(score).to(self.score)
        return self.score


class BERTScore(TorchBERTScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.scores = []

    def update(self, preds: List[str], target: List[str]) -> None:
        super().update(preds, target)
        self.preds_input_ids[-1] = self.preds_input_ids[-1].to(self.device)
        self.preds_attention_mask[-1] = self.preds_attention_mask[-1].to(
            self.device
        )
        self.target_input_ids[-1] = self.target_input_ids[-1].to(self.device)
        self.target_attention_mask[-1] = self.target_attention_mask[-1].to(
            self.device
        )

    def _get_input_dict(
        self, input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output_dict = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
        }
        return output_dict

    @cat_states
    def compute(self) -> Dict[str, Union[List[float], str]]:
        """Calculate BERT scores.

        Return:
            Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.
        """
        if len(self.preds_input_ids) == 0:
            return None
        result = bert_score(
            preds=self._get_input_dict(
                self.preds_input_ids, self.preds_attention_mask
            ),
            target=self._get_input_dict(
                self.target_input_ids, self.target_attention_mask
            ),
            model_name_or_path=self.model_name_or_path,
            num_layers=self.num_layers,
            all_layers=self.all_layers,
            model=self.model,
            user_tokenizer=self.tokenizer if self.user_tokenizer else None,
            user_forward_fn=self.user_forward_fn,
            verbose=self.verbose,
            idf=self.idf,
            device=self.embedding_device,
            max_length=self.max_length,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            return_hash=self.return_hash,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            baseline_path=self.baseline_path,
            baseline_url=self.baseline_url,
        )
        self.scores = result["f1"]
        self.score = torch.sum(torch.tensor(result["f1"]).to(self.score.device))
        self.score_count = torch.tensor(len(result["f1"])).to(
            self.score_count.device
        )
        return self.score / self.score_count


class Perplexity(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "loss_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, loss: torch.Tensor) -> None:
        self.loss += loss
        self.loss_count += 1

    def compute(self) -> torch.Tensor:
        return (
            torch.exp(self.loss / self.loss_count) if self.loss_count else None
        )


class TTCriterion(nn.Module):
    """Transformation Telling Criterion.

    The outputs should be a dictionary of torch.Tensor with the following keys:
    - "logits": the logits of the model
    - "label_ids": the targets captions
    - "context": the context, required by TellingLoss
    """

    def __init__(self, loss=None, bert_score_model="roberta-large") -> None:
        super().__init__()

        self.tokenizer = SimpleTokenizer()

        self.loss = loss

        self.bleu_4 = BLEUScore(n_gram=4)
        self.rouge = ROUGE()
        self.meteor = METEOR()
        self.cider = CIDEr()
        self.bert_score = BERTScore(model_name_or_path=bert_score_model)
        self.perplexity = Perplexity()

        self.train_metrics = [
            ("PPL", self.perplexity),
        ]
        self.eval_metrics = [
            ("BLEU_4", self.bleu_4),
            ("ROUGE", self.rouge),
            ("METEOR", self.meteor),
            ("CIDEr", self.cider),
            ("BERTScore", self.bert_score),
        ]

    def forward(self, outputs, eval=False):
        result = self.loss(outputs, return_dict=True)
        self.perplexity.update(result["loss"])
        if eval:
            preds = rearrange(
                outputs["logits"].argmax(dim=-1), "B N L -> (B N) L"
            )
            target = rearrange(outputs["label_ids"], "B N L -> (B N) L")
            mask = rearrange(outputs["label_mask"].sum(-1) > 0, "B N -> (B N)")
            preds = [
                self.tokenizer.smart_decode(tolist(x)) for x in preds[mask]
            ]
            target = [
                self.tokenizer.smart_decode(tolist(x)) for x in target[mask]
            ]
            self.update(preds, target)
        outputs.update(result)
        return outputs

    def update(self, preds, target):
        eval_targets = [
            (p, f) for p, f in zip(preds, target) if len(p) > 0 or len(f) > 0
        ]
        preds, target = list(zip(*eval_targets))
        preds, target = list(preds), list(target)
        for metric_name, metric in self.eval_metrics:
            if metric_name in ["BLEU_4"]:
                metric.update(preds, [target])
            else:
                metric.update(preds, target)

    def compute(self, verbose=False):
        metrics = {}
        for metric_name, metric in self.train_metrics + self.eval_metrics:
            if verbose:
                value, time_cost = with_time(metric.compute, pretty_time=True)()
                logger.info(f"- {metric_name}: {value} ({time_cost})")
            else:
                value = metric.compute()
            if value is not None:
                metrics[metric_name] = value
        return metrics

    def reset(self):
        for _, metric in self.eval_metrics:
            metric.reset()

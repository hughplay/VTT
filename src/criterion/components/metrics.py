import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torchmetrics import Accuracy as AccuracyMetric
from torchmetrics import Metric
from torchmetrics.metric import jit_distributed_available
from torchmetrics.utilities.data import dim_zero_cat
from transformers import logging as transformers_logging

from src.dataset.text import SimpleTokenizer
from src.utils.arraytool import tolist

from .pycocoevalcap.cider.cider import CiderScorer as coco_cider
from .pycocoevalcap.meteor.meteor import Meteor as coco_meteor
from .pycocoevalcap.rouge.rouge import Rouge as coco_rouge
from .pycocoevalcap.spice.spice import Spice as coco_spice
from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# disable the warning from huggingface:
# The current process just got forked, after parallelism has already been used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# disable the warning from huggingface:
# Some weights of the model checkpoint at roberta-large were not used when
# initializing RobertaModel
transformers_logging.set_verbosity_error()

# current dir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROBERTA_BASELINE = os.path.join(
    CURRENT_DIR, "roberta_large_baseline.tsv"
)


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


class BLEU(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n_gram=4, smooth=True, **kwargs):
        super().__init__(**kwargs)

        self.n_gram = n_gram
        self.smooth = smooth
        self.tokenizer = PTBTokenizer()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:
        preds = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(preds))
        )
        target = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(target))
        )
        for pred, target in zip(preds, target):
            score = torch.tensor(
                sentence_bleu(
                    [target.split()],
                    pred.split(),
                    smoothing_function=SmoothingFunction().method7,
                )
            ).to(self.score)
            self.scores.append(score)
            self.score_count += 1
            self.score += score

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
        _, scores = self.rouge.compute_score(target, preds)
        self.score += sum(scores)
        self.scores.append(torch.tensor(scores).to(self.score.device))
        self.score_count += len(scores)

    @cat_states
    def compute(self):
        return self.score / self.score_count if self.score_count else None


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
        _, scores = self.meteor.compute_score(target, preds)
        self.score += sum(scores)
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
                torch.tensor(len(pred)).to(self.score.device)
            )
            self.preds.append(torch.tensor(pred).to(self.score.device))
        target = coco_extract_sequences(
            self.tokenizer.tokenize(coco_fmt_sequences(target))
        )
        for t in target:
            t = self.clip_tokenizer.encode(t)
            self.target_offset.append(
                torch.tensor(len(t)).to(self.score.device)
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


class SPICE(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spice = coco_spice()
        self.tokenizer = PTBTokenizer()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "score_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:
        preds = self.tokenizer.tokenize(coco_fmt_sequences(preds))
        target = self.tokenizer.tokenize(coco_fmt_sequences(target))
        _, scores = self.spice.compute_score(target, preds)
        scores = [item["All"]["f"] for item in scores]
        self.score += sum(scores)
        self.scores.append(torch.tensor(scores).to(self.score.device))
        self.score_count += len(scores)

    @cat_states
    def compute(self):
        return self.score / self.score_count if self.score_count else None


class BERTScore(Metric):

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        model_name_or_path: str = None,
        num_layers=17,
        rescale_with_baseline=True,
        baseline_path=DEFAULT_ROBERTA_BASELINE,
        idf=False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model_name_or_path = model_name_or_path
        self.idf = idf
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        self.num_layers = num_layers

        self.clip_tokenizer = SimpleTokenizer()

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds_offset", default=[], dist_reduce_fx="cat")
        self.add_state("target_offset", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.scores = []

    def update(self, preds: Sequence[str], target: Sequence[str]) -> None:

        for pred in preds:
            pred = self.clip_tokenizer.encode(pred)
            self.preds_offset.append(
                torch.tensor(len(pred)).to(self.score.device)
            )
            self.preds.append(torch.tensor(pred).to(self.score.device))
        for t in target:
            t = self.clip_tokenizer.encode(t)
            self.target_offset.append(
                torch.tensor(len(t)).to(self.score.device)
            )
            self.target.append(torch.tensor(t).to(self.score.device))

    @cat_states
    def compute(self):
        if len(self.preds) == 0:
            return None
        start_p = 0
        start_t = 0
        preds, target = [], []
        for offset_p, offset_t in zip(self.preds_offset, self.target_offset):
            p = self.clip_tokenizer.decode(
                tolist(self.preds[start_p : start_p + offset_p])
            )
            t = self.clip_tokenizer.decode(
                tolist(self.target[start_t : start_t + offset_t])
            )
            preds.append(p)
            target.append(t)
            start_p += offset_p
            start_t += offset_t
        _, _, f1 = bert_score(
            preds,
            target,
            model_type=self.model_name_or_path,
            num_layers=self.num_layers,
            rescale_with_baseline=self.rescale_with_baseline,
            baseline_path=self.baseline_path,
            idf=self.idf,
            lang="en",
            device=self.device,
        )
        self.scores = tolist(f1)
        self.score = torch.tensor(self.scores).mean().to(self.score)
        return self.score


class Perplexity(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

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


class Accuracy(AccuracyMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        average: Optional[str] = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        subset_accuracy: bool = False,
        classes: Sequence[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold,
            num_classes,
            average,
            mdmc_average,
            ignore_index,
            top_k,
            multiclass,
            subset_accuracy,
            **kwargs,
        )
        self.classes = classes
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds, target)
        self.preds.extend(preds)
        self.target.extend(target)

    @property
    def str_preds(self) -> List[str]:
        return [self.classes[x] for x in tolist(self.preds)]

    @property
    def str_target(self) -> List[str]:
        return [self.classes[x] for x in tolist(self.target)]

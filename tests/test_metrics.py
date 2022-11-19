import logging

import pytest
import torch

from src.criterion.tell import TTCriterion

logger = logging.getLogger(__name__)


def test_metrics():
    ttc = TTCriterion(
        bert_score_model="/data/pretrain/transformers/roberta-large"
    )
    ttc.update(
        ["the cat is on the mat", "this is my world"],
        ["the cat is on the mat", "this is your world"],
    )
    ttc.update(
        ["the dog is on the mat", ""],
        ["the dog is not on the mat", ""],
    )
    ttc.update(
        ["hello world"],
        ["what a sunny day"],
    )

    result = ttc.compute(verbose=True)

    for key, val in ttc.scores.items():
        logger.info(f"{key}: {val}")

    assert "BLEU_4" in result
    assert "ROUGE" in result
    assert "METEOR" in result
    assert "CIDEr" in result
    assert "SPICE" in result
    assert "BERTScore" in result

    # the empty string should be ignored
    assert ttc.rouge.score_count.item() == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_metrics_cuda():
    ttc = TTCriterion(
        bert_score_model="/data/pretrain/transformers/roberta-large"
    ).cuda()
    ttc.update(
        ["the cat is on the mat", "this is my world"],
        ["the cat is on the mat", "this is your world"],
    )
    ttc.update(
        ["the dog is on the mat", ""],
        ["the dog is not on the mat", ""],
    )
    ttc.update(
        ["hello world"],
        ["what a sunny day"],
    )

    result = ttc.compute(verbose=True)

    for key, val in ttc.scores.items():
        logger.info(f"{key}: {val}")

    assert "BLEU_4" in result
    assert "ROUGE" in result
    assert "METEOR" in result
    assert "CIDEr" in result
    assert "SPICE" in result
    assert "BERTScore" in result

    # the empty string should be ignored
    assert ttc.rouge.score_count.item() == 4

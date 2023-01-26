import logging

import pytest
import torch

from src.criterion.text import TextCriterion

logger = logging.getLogger(__name__)


def test_metrics():
    textc = TextCriterion(
        bert_score_model="/data/pretrain/transformers/roberta-large"
    )
    textc.update(
        ["the cat is on the mat", "this is my world"],
        ["the cat is on the mat", "this is your world"],
    )
    textc.update(
        ["the dog is on the mat", ""],
        ["the dog is not on the mat", ""],
    )
    textc.update(
        ["hello world"],
        ["what a sunny day"],
    )

    result = textc.compute(verbose=True)

    for key, val in textc.scores.items():
        logger.info(f"{key}: {val}")

    assert "BLEU_4" in result
    assert "ROUGE" in result
    assert "METEOR" in result
    assert "CIDEr" in result
    assert "SPICE" in result
    assert "BERTScore" in result

    # the empty string should be ignored
    assert textc.rouge.score_count.item() == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_metrics_cuda():
    textc = TextCriterion(
        bert_score_model="/data/pretrain/transformers/roberta-large"
    ).cuda()
    textc.update(
        ["the cat is on the mat", "this is my world"],
        ["the cat is on the mat", "this is your world"],
    )
    textc.update(
        ["the dog is on the mat", ""],
        ["the dog is not on the mat", ""],
    )
    textc.update(
        ["hello world"],
        ["what a sunny day"],
    )

    result = textc.compute(verbose=True)

    for key, val in textc.scores.items():
        logger.info(f"{key}: {val}")

    assert "BLEU_4" in result
    assert "ROUGE" in result
    assert "METEOR" in result
    assert "CIDEr" in result
    assert "SPICE" in result
    assert "BERTScore" in result

    # the empty string should be ignored
    assert textc.rouge.score_count.item() == 4

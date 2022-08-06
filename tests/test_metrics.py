import torch

from src.criterion.tell import TTCriterion


def test_metrics():
    ttc = TTCriterion()
    ttc.update(
        ["the cat is on the mat", "this is my world"],
        ["the cat is on the mat", "this is your world"],
    )
    result = ttc.compute()

    assert "BLEU_4" in result
    assert "ROUGE" in result
    assert "METEOR" in result
    assert "CIDEr" in result
    assert "BERTScore" in result

    for name, value in result.items():
        assert isinstance(value, torch.Tensor)
        print(f"{name}: {value}")

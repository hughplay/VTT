import pytest
import torch

from src.criterion.loss import GenerationLoss, TellingLossV1
from src.dataset.vtt import CATEGORIES, TOPICS
from src.model.components.text_decoder import N_WORDS


@pytest.mark.parametrize("Loss", [GenerationLoss, TellingLossV1])
def test_loss(Loss):

    # prepare inputs
    batch_size = 2
    max_transformations = 12
    max_states = max_transformations + 1
    max_words = 24
    dim = 512
    label_ids = torch.randint(
        0, N_WORDS, (batch_size, max_transformations, max_words)
    )
    label_mask = torch.ones(batch_size, max_transformations, max_words).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_states))
    for i in range(batch_size):
        for j in range(max_transformations):
            label_mask[i, j, : rand_len[i, j]] = True
    context = torch.randn(batch_size, max_states, dim)
    logits = torch.ones(batch_size, max_transformations, max_words, N_WORDS)
    category = torch.randint(0, len(CATEGORIES), (batch_size,))
    topic = torch.randint(0, len(TOPICS), (batch_size,))
    inputs = {
        "context": context,
        "label_ids": label_ids,
        "label_mask": label_mask,
        "logits": logits,
        "category": category,
        "topic": topic,
    }

    # check output
    loss = Loss()
    result = loss(inputs, return_dict=True)

    assert "loss" in result
    assert result["loss"].size() == torch.Size([])

    estimate_gen_loss = -torch.log(torch.tensor(1 / N_WORDS))
    print()
    print(f"Estimated Gen loss: {estimate_gen_loss.item()}")

    if Loss is GenerationLoss:

        print(f"Result Gen loss: {result['loss'].item()}")
        assert result["loss"].item() == pytest.approx(
            estimate_gen_loss.item(), rel=0.01
        )

    elif Loss is TellingLossV1:

        assert "generation_loss" in result
        print(f"Result Gen loss: {result['generation_loss'].item()}")
        assert result["generation_loss"].item() == pytest.approx(
            estimate_gen_loss.item(), rel=0.01
        )

        assert "classification_loss" in result
        assert "category_logits" in result
        assert result["category_logits"].size() == torch.Size(
            [batch_size, len(CATEGORIES)]
        )
        assert "category_loss" in result
        estimate_cat_loss = -torch.log(torch.tensor(1 / len(CATEGORIES)))
        print(f"Estimated Cat loss: {estimate_cat_loss.item()}")
        print(f"Result Cat loss: {result['category_loss'].item()}")
        assert result["category_loss"].item() == pytest.approx(
            estimate_cat_loss.item(), rel=1.0
        )
        assert "topic_logits" in result
        assert result["topic_logits"].size() == torch.Size(
            [batch_size, len(TOPICS)]
        )
        assert "topic_loss" in result
        estimate_topic_loss = -torch.log(torch.tensor(1 / len(TOPICS)))
        print(f"Estimated Topic loss: {estimate_topic_loss.item()}")
        print(f"Result Topic loss: {result['topic_loss'].item()}")
        assert result["topic_loss"].item() == pytest.approx(
            estimate_topic_loss.item(), rel=1.0
        )
        assert "construction_loss" in result

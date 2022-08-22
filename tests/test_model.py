import pytest
import torch

from src.model.components.image_encoder import _INPUT_OUTPUT_DIM
from src.model.components.text_decoder import N_WORDS
from src.model.cst import CST
from src.model.glacnet import GLACNet
from src.model.ttnet import TTNet


@pytest.mark.parametrize("Model", [CST, GLACNet, TTNet])
def test_model(Model):

    # prepare model
    model = Model()

    # prepare inputs
    batch_size = 2
    img_dim = _INPUT_OUTPUT_DIM[model.image_encoder.name]["input"]
    max_transformations = 12
    max_states = max_transformations + 1
    max_words = 24
    # inputs: images
    states = torch.randn(batch_size, max_states, 3, img_dim, img_dim)
    states_mask = torch.zeros(batch_size, max_states).bool()
    rand_len = torch.randint(1, max_states, (batch_size,))
    for i in range(batch_size):
        states_mask[i, : rand_len[i]] = True
    # inputs: captions
    label_ids = torch.randint(
        0, N_WORDS, (batch_size, max_transformations, max_words)
    )
    label_mask = torch.zeros(batch_size, max_transformations, max_words).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_states))
    for i in range(batch_size):
        for j in range(max_transformations):
            label_mask[i, j, : rand_len[i, j]] = True

    # check output
    output = model(states, states_mask, label_ids, label_mask)
    assert output["logits"].shape == (
        batch_size,
        max_transformations,
        max_words,
        N_WORDS,
    )

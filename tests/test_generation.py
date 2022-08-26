import pytest
import torch

from src.model.components.image_encoder import _INPUT_OUTPUT_DIM
from src.model.components.text_decoder import N_WORDS
from src.model.cst import CST
from src.model.glacnet import GLACNet
from src.model.ttnet import TTNet


@pytest.mark.parametrize("Model", [CST, GLACNet, TTNet])
def test_greedy(Model):

    max_words = 24
    # prepare model
    model = Model(
        generate_cfg={
            "do_sample": False,
            "top_p": 0.0,
            "top_k": 0,
            "temperature": 1.0,
            "max_words": max_words,
        }
    )

    # prepare inputs
    batch_size = 2
    img_dim = _INPUT_OUTPUT_DIM[model.image_encoder.name]["input"]
    max_transformations = 12
    max_states = max_transformations + 1
    # inputs: images
    states = torch.randn(batch_size, max_states, 3, img_dim, img_dim)
    states_mask = torch.zeros(batch_size, max_states).bool()
    rand_len = torch.randint(1, max_states, (batch_size,))
    for i in range(batch_size):
        states_mask[i, : rand_len[i]] = True

    # test on gpu
    if torch.cuda.is_available():
        model.cuda()
        states = states.cuda()
        states_mask = states_mask.cuda()

    # check output
    output = model(states, states_mask)
    assert output["sequence"].shape == (
        batch_size,
        max_transformations,
        max_words,
    )


@pytest.mark.parametrize("Model", [CST, GLACNet, TTNet])
def test_top_k_top_p(Model):

    max_words = 24
    # prepare model
    model = Model(
        generate_cfg={
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 1000,
            "temperature": 1.0,
            "max_words": max_words,
        }
    )

    # prepare inputs
    batch_size = 2
    img_dim = _INPUT_OUTPUT_DIM[model.image_encoder.name]["input"]
    max_transformations = 12
    max_states = max_transformations + 1
    # inputs: images
    states = torch.randn(batch_size, max_states, 3, img_dim, img_dim)
    states_mask = torch.zeros(batch_size, max_states).bool()
    rand_len = torch.randint(1, max_states, (batch_size,))
    for i in range(batch_size):
        states_mask[i, : rand_len[i]] = True

    # test on gpu
    if torch.cuda.is_available():
        model.cuda()
        states = states.cuda()
        states_mask = states_mask.cuda()

    # check output
    output = model(states, states_mask)
    assert output["sequence"].shape == (
        batch_size,
        max_transformations,
        max_words,
    )

from itertools import product

import pytest
import torch

from src.model.components.text_decoder import (
    N_WORDS,
    ContextLSTMText,
    IndependentLSTMText,
    TransformerText,
)


@pytest.mark.parametrize(
    "num_layers,state_dim,feature_dim,word_emb_dim",
    product([1, 2], [64, 128], [64, 128], [64, 128]),
)
def test_independent_lstm(num_layers, state_dim, feature_dim, word_emb_dim):

    # prepare inputs
    batch_size = 2
    max_image = 12
    max_words = 24
    hn = torch.randn(num_layers, batch_size, state_dim)
    cn = torch.randn(num_layers, batch_size, state_dim)
    state = (hn, cn)
    features = torch.randn(batch_size, max_image, feature_dim)
    label_ids = torch.randint(0, N_WORDS, (batch_size, max_image, max_words))
    mask = torch.zeros(batch_size, max_image, max_words).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_image))
    for i in range(batch_size):
        for j in range(max_image):
            mask[i, j, : rand_len[i, j]] = True

    # prepare model
    hidden_dim = word_emb_dim
    decoder = IndependentLSTMText(
        state_dim=state_dim,
        feature_dim=feature_dim,
        max_seq_len=max_image,
        word_emb_dim=word_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if torch.cuda.is_available():
        decoder.cuda()
        state = (hn.cuda(), cn.cuda())
        features = features.cuda()
        label_ids = label_ids.cuda()
        mask = mask.cuda()

    # get output
    output = decoder(state, features, label_ids, mask)

    # check output
    assert output.shape == (batch_size, max_image, max_words, N_WORDS)


@pytest.mark.parametrize(
    "num_layers,context_dim,word_emb_dim",
    product([1, 2], [64, 128], [64, 128]),
)
def test_context_lstm(num_layers, context_dim, word_emb_dim):

    # prepare inputs
    batch_size = 2
    max_image = 12
    max_words = 24
    context = torch.randn(batch_size, max_image, context_dim)
    label_ids = torch.randint(0, N_WORDS, (batch_size, max_image, max_words))
    mask = torch.zeros(batch_size, max_image, max_words).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_image))
    for i in range(batch_size):
        for j in range(max_image):
            mask[i, j, : rand_len[i, j]] = True

    # prepare model
    hidden_dim = word_emb_dim
    decoder = ContextLSTMText(
        context_dim=context_dim,
        word_emb_dim=word_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if torch.cuda.is_available():
        decoder.cuda()
        context = context.cuda()
        label_ids = label_ids.cuda()
        mask = mask.cuda()

    # get output
    output = decoder(context, label_ids, mask)

    # check output
    assert output.shape == (batch_size, max_image, max_words, N_WORDS)


@pytest.mark.parametrize(
    "context_dim,hidden_dim,position_embedding",
    product(
        [64, 128], [64, 128], ["fixed", "absolute", "infused_fixed", "relative"]
    ),
)
def test_transformer(context_dim, hidden_dim, position_embedding):

    # prepare inputs
    batch_size = 2
    max_image = 12
    max_words = 24
    context = torch.randn(batch_size, max_image, context_dim)
    label_ids = torch.randint(0, N_WORDS, (batch_size, max_image, max_words))
    mask = torch.zeros(batch_size, max_image, max_words).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_image))
    for i in range(batch_size):
        for j in range(max_image):
            mask[i, j, : rand_len[i, j]] = True

    # prepare model
    decoder = TransformerText(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        position_embedding=position_embedding,
        max_words=max_words,
    )

    if torch.cuda.is_available():
        decoder.cuda()
        context = context.cuda()
        label_ids = label_ids.cuda()
        mask = mask.cuda()

    # get output
    output = decoder(context, label_ids, mask)

    # check output
    assert output.shape == (batch_size, max_image, max_words, N_WORDS)

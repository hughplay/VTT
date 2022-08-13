import pytest
import torch

from src.model.components.context_encoder import (
    GLocalContext,
    SimpleLSTMContext,
    TransformerContext,
)


@pytest.mark.parametrize(
    "num_layers, bidirectional", [(1, False), (1, True), (2, False), (2, True)]
)
def test_simple_lstm(num_layers, bidirectional):

    # prepare inputs
    batch_size = 2
    max_image = 12
    feat_dim = 128
    x = torch.randn(batch_size, max_image, feat_dim)
    mask = torch.zeros(batch_size, max_image).bool()
    rand_len = torch.randint(1, max_image, (batch_size,))
    for i in range(batch_size):
        mask[i, : rand_len[i]] = True

    # prepare model
    hidden_dim = 64
    encoder = SimpleLSTMContext(
        input_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    if torch.cuda.is_available():
        encoder.cuda()
        x = x.cuda()
        mask = mask.cuda()

    # get output
    output = encoder(x, mask)

    # check output
    assert output["context"].shape == (
        batch_size,
        max_image,
        hidden_dim * (2 if bidirectional else 1),
    )

    hn, cn = output["state"]
    assert hn.shape == (
        num_layers * (2 if bidirectional else 1),
        batch_size,
        hidden_dim,
    )
    assert cn.shape == (
        num_layers * (2 if bidirectional else 1),
        batch_size,
        hidden_dim,
    )


@pytest.mark.parametrize(
    "num_layers, bidirectional", [(1, False), (1, True), (2, False), (2, True)]
)
def test_glocal(num_layers, bidirectional):

    # prepare inputs
    batch_size = 2
    max_image = 12
    feat_dim = 128
    x = torch.randn(batch_size, max_image, feat_dim)
    mask = torch.zeros(batch_size, max_image).bool()
    rand_len = torch.randint(1, max_image, (batch_size,))
    for i in range(batch_size):
        mask[i, : rand_len[i]] = True

    # prepare model
    hidden_dim = 64
    encoder = GLocalContext(
        input_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    if torch.cuda.is_available():
        encoder.cuda()
        x = x.cuda()
        mask = mask.cuda()

    # get output
    output = encoder(x, mask)

    # check output
    assert output["context"].shape == (batch_size, max_image, hidden_dim)

    hn, cn = output["state"]
    assert hn.shape == (
        num_layers * (2 if bidirectional else 1),
        batch_size,
        hidden_dim,
    )
    assert cn.shape == (
        num_layers * (2 if bidirectional else 1),
        batch_size,
        hidden_dim,
    )


@pytest.mark.parametrize(
    "position_embedding", ["fixed", "absolute", "infused_fixed", "relative"]
)
def test_transformer(position_embedding):

    # prepare inputs
    batch_size = 2
    max_image = 12
    feat_dim = 128
    x = torch.randn(batch_size, max_image, feat_dim)
    mask = torch.zeros(batch_size, max_image).bool()
    rand_len = torch.randint(1, max_image, (batch_size,))
    for i in range(batch_size):
        mask[i, : rand_len[i]] = True

    # prepare model
    heads = 8
    num_layers = 2
    encoder = TransformerContext(
        input_dim=feat_dim,
        num_layers=num_layers,
        heads=heads,
        position_embedding=position_embedding,
        max_seq_len=max_image,
    )

    if torch.cuda.is_available():
        encoder.cuda()
        x = x.cuda()
        mask = mask.cuda()

    # get output
    output = encoder(x, mask)

    # check output
    assert output["context"].shape == (batch_size, max_image, feat_dim)

    attn = output["attention"]
    assert len(attn) == num_layers
    assert attn[0].shape == (batch_size, heads, max_image, max_image)

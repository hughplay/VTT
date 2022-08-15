import pytest
import torch

from src.model.components.image_encoder import (
    _INPUT_OUTPUT_DIM,
    ImageEncoder,
    available_models,
)


@pytest.mark.parametrize("name", available_models)
def test_image_encoders(name):
    promise_input_dim = _INPUT_OUTPUT_DIM[name]["input"]

    output_dim = 512
    encoder = ImageEncoder(
        name,
        # pretrained=False,
        pretrained=True,
        promise_input_dim=promise_input_dim,
        output_dim=output_dim,
    )
    if torch.cuda.is_available():
        encoder.cuda()
    encoder.eval()

    batch_size = 1
    channel = 3
    num_seq = 2

    images = torch.randn(
        batch_size, channel, promise_input_dim, promise_input_dim
    )
    if torch.cuda.is_available():
        images = images.cuda()
    out = encoder(images)

    assert out.shape == (batch_size, output_dim)
    print(f"{name}")
    print(f"- input dimension: {promise_input_dim}")
    print(f"- encoder output dimension: {encoder.encoder.output_dim}")
    print(f"- final output dimension: {output_dim}")
    print(f"- skip connect: {encoder.skip_connect}")

    list_images = torch.randn(
        batch_size, num_seq, channel, promise_input_dim, promise_input_dim
    )
    if torch.cuda.is_available():
        list_images = list_images.cuda()
    out = encoder(list_images)
    assert out.shape == (batch_size, num_seq, output_dim)

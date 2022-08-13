import pytest
import torch

from src.model.components.image_encoder import ImageEncoder, available_models


@pytest.mark.parametrize("name", available_models)
def test_image_encoders(name):
    if name == "RN50x4":
        promise_input_dim = 288
    elif name == "RN50x16":
        promise_input_dim = 384
    elif name == "inception_v3":
        promise_input_dim = 299
    else:
        promise_input_dim = 224

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

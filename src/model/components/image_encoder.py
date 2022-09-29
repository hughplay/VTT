"""Uniform image encoder API.

Args:
    name: name of the model to use
    pretrained: whether to use pretrained weights
    promise_input_dim: the dimension of the input images
    fine_tune: whether to fine tune the model
    num_classes: number of classes in the dataset
    output_dim: the dimension of the output

Input:
    images: tensor of shape (*, 3, promise_input_dim, promise_input_dim), where
        * is the number of images of arbitrary shapes.

Output:
    output: tensor of shape (*, output_dim)

Available encoders:
    - ResNets: resnet18, resnet34, resnet50, resnet101, resnet152
    - InceptionNet V3: inception_v3
    - CLIP: ViT-B/32, ViT-B/16, RN50, RN101, RN50x4, RN50x16
"""
import logging
from functools import wraps
from itertools import chain

import timm
import torch
from torch import nn
from torchvision.models import inception, resnet

from src.model.components.clip import clip

logger = logging.getLogger(__name__)

# fmt: off
_MODELS = {
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "inception_v3": ["inception_v3"],
    "clip": ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
    "beit": ["beit_large_patch16_224"],
    "swin": ["swin_large_patch4_window7_224"],
    "vit": ["vit_large_patch16_224"]
}
# chain _MODEL values
available_models = list(chain(*_MODELS.values()))
# fmt: on

_INPUT_OUTPUT_DIM = {
    "resnet18": {"input": 224, "output": 512},
    "resnet34": {"input": 224, "output": 512},
    "resnet50": {"input": 224, "output": 512},
    "resnet101": {"input": 224, "output": 2048},
    "resnet152": {"input": 224, "output": 2048},
    "inception_v3": {"input": 299, "output": 2048},
    "RN50": {"input": 224, "output": 1024},
    "RN101": {"input": 224, "output": 512},
    "RN50x4": {"input": 288, "output": 640},
    "RN50x16": {"input": 384, "output": 768},
    "RN50x64": {"input": 448, "output": 1024},
    "ViT-B/32": {"input": 224, "output": 512},
    "ViT-B/16": {"input": 224, "output": 512},
    "ViT-L/14": {"input": 224, "output": 768},
    "ViT-L/14@336px": {"input": 336, "output": 768},
    "beit_large_patch16_224": {"input": 224, "output": 1024},
    "swin_large_patch4_window7_224": {"input": 224, "output": 1536},
    "vit_large_patch16_224": {"input": 224, "output": 1024},
}


def infer_output_dim(model, height, width):
    batch, channel = 1, 3
    tensor = torch.zeros(batch, channel, height, width).to(
        next(model.parameters()).device
    )
    model.eval()
    dim_flat = torch.prod(torch.tensor(model(tensor).shape[1:])).item()
    return dim_flat


def dim_agnostic_encode(func):
    """Encode images with model from shape ([?], C, H, W) to ([?], d)"""

    @wraps(func)
    def wrapper(self, images):
        flat_dim, keep_dim = list(images.shape[:-3]), list(images.shape[-3:])
        pre_encoder_dim = [-1] + keep_dim
        post_encoder_dim = flat_dim + [-1]

        out = images.view(pre_encoder_dim)
        out = func(self, out)
        out = out.view(post_encoder_dim)

        return out

    return wrapper


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        promise_input_dim: int = 224,
    ):
        """ResNet image encoder.

        |           | Input resolution | Embedding dimension |
        |-----------|------------------|---------------------|
        | resnet18  |              224 |                 512 |
        | resnet34  |              224 |                 512 |
        | resnet50  |              224 |                 512 |
        | resnet101 |              224 |                2048 |
        | resnet152 |              224 |                2048 |
        """
        super().__init__()

        model = getattr(resnet, name)(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.output_dim = infer_output_dim(
            self.resnet, promise_input_dim, promise_input_dim
        )

    def forward(self, images):
        return self.resnet(images).view(images.size(0), -1)


class InceptionEncoder(nn.Module):
    def __init__(
        self,
        name: str = "inception_v3",
        pretrained: bool = True,
        promise_input_dim: int = 299,
    ):
        """InceptionNet image encoder.

        |              | Input resolution | Embedding dimension |
        |--------------|------------------|---------------------|
        | inception_v3 |              299 |                2048 |
        """
        super().__init__()

        assert promise_input_dim == 299, (
            "InceptionV3 only supports 299x299. "
            "Reference: https://arxiv.org/pdf/1512.00567v3.pdf."
        )

        self.inception = getattr(inception, name)(
            init_weights=pretrained, aux_logits=False
        )
        self.inception.fc = nn.Identity()

        self.output_dim = infer_output_dim(
            self.inception, promise_input_dim, promise_input_dim
        )

    def forward(self, images):
        return self.inception(images)


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        name: str = "ViT-B/32",
        pretrained: bool = True,
        promise_input_dim: int = 224,
    ):
        """CLIP image encoder.

        |                | Input resolution | Embedding dimension |
        |----------------|------------------|---------------------|
        | RN50           |              224 |                1024 |
        | RN101          |              224 |                 512 |
        | RN50x4         |              288 |                 640 |
        | RN50x16        |              384 |                 768 |
        | RN50x64        |              448 |                1024 |
        | ViT-B/32       |              224 |                 512 |
        | ViT-B/16       |              224 |                 512 |
        | ViT-L/14       |              224 |                 768 |
        | ViT-L/14@336px |              336 |                 768 |
        """
        super().__init__()

        assert name in _MODELS["clip"]
        assert (
            promise_input_dim == _INPUT_OUTPUT_DIM[name]["input"]
        ), f"input dim must be {_INPUT_OUTPUT_DIM[name]['input']} for {name}"

        model, _ = clip.load(
            name, pretrained=pretrained, convert_fp16=False, device="cpu"
        )
        self.clip_visual = model.visual
        self.output_dim = self.clip_visual.output_dim

    def forward(self, images):
        return self.clip_visual(images)


class PretrainedEncoder(nn.Module):
    def __init__(
        self,
        name: str = "beit_large_patch16_224",
        pretrained: bool = True,
        promise_input_dim: int = 224,
    ):
        """Other ImageNet pretrained image encoders.

        |                               | Input resolution | Embedding dimension |
        |-------------------------------|------------------|---------------------|
        | beit_large_patch16_224        |              224 |                1024 |
        | swin_large_patch4_window7_224 |              224 |                1536 |
        | vit_large_patch16_224         |              224 |                1024 |
        """
        super().__init__()

        assert name in timm.list_models()
        assert (
            promise_input_dim == _INPUT_OUTPUT_DIM[name]["input"]
        ), f"input dim must be {_INPUT_OUTPUT_DIM[name]['input']} for {name}"

        self.model = timm.create_model(
            name, pretrained=pretrained, num_classes=0
        )
        self.output_dim = self.model.num_features

    def forward(self, images):
        return self.model(images)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        promise_input_dim: int = None,
        finetune: bool = False,
        output_dim: int = 512,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.promise_input_dim = (
            promise_input_dim
            if promise_input_dim is not None
            else _INPUT_OUTPUT_DIM[name]["input"]
        )
        if name not in available_models:
            raise ValueError(f"{name} is not available")
        # elif name in _MODELS["resnet"]:
        #     self.encoder = ResNetEncoder(
        #         self.name, self.pretrained, self.promise_input_dim
        #     )
        # elif name in _MODELS["inception_v3"]:
        #     self.encoder = InceptionEncoder(
        #         self.name, self.pretrained, self.promise_input_dim
        #     )
        elif name in _MODELS["clip"]:
            self.encoder = CLIPEncoder(
                self.name, self.pretrained, self.promise_input_dim
            )
        else:
            self.encoder = PretrainedEncoder(
                self.name, self.pretrained, self.promise_input_dim
            )

        self.finetune = finetune
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.output_dim = output_dim
        self.linear = nn.Linear(self.encoder.output_dim, output_dim)
        self.bn = (
            nn.BatchNorm1d(output_dim, momentum=0.01)
            if batch_norm
            else nn.Identity()
        )

        # skip connect if output_dim of encoder and linear are equal
        self.skip_connect = self.encoder.output_dim == output_dim

    @dim_agnostic_encode
    def forward(self, images):
        if not self.finetune:
            self.encoder.eval()
        en_output = self.encoder(images)

        output = self.linear(en_output)
        output = self.bn(output)

        if self.skip_connect:
            output = output + en_output

        return output


class EarlyDiffImageEncoder(ImageEncoder):
    def forward(self, states, states_mask):

        B, N, C, H, W = states.size()

        features = super().forward(states)

        shifted_states = torch.zeros_like(states)
        shifted_states[:, 1:, ...] = states[:, :-1, ...]
        last_states_pos = torch.sum(states_mask, dim=-1) - 1
        shifted_states[:, 0, ...] = states[torch.arange(B), last_states_pos]

        diff = states - shifted_states
        diff_features = super().forward(diff)

        return features, diff_features


class LateDiffImageEncoder(ImageEncoder):
    def forward(self, states, states_mask):

        B, N, C, H, W = states.size()

        features = super().forward(states)

        shifted_features = torch.zeros_like(features)
        shifted_features[:, 1:, ...] = features[:, :-1, ...]
        last_states_pos = torch.sum(states_mask, dim=-1) - 1
        shifted_features[:, 0, ...] = features[torch.arange(B), last_states_pos]

        diff_features = features - shifted_features

        return features, diff_features


class BiDiffImageEncoder(ImageEncoder):
    def forward(self, states, states_mask):

        B, N, C, H, W = states.size()

        features = super().forward(states)
        last_states_pos = torch.sum(states_mask, dim=-1) - 1

        shifted_states = torch.zeros_like(states)
        shifted_states[:, 1:, ...] = states[:, :-1, ...]
        shifted_states[:, 0, ...] = states[torch.arange(B), last_states_pos]

        early_diff = states - shifted_states
        early_diff_features = super().forward(early_diff)

        shifted_features = torch.zeros_like(features)
        shifted_features[:, 1:, ...] = features[:, :-1, ...]
        shifted_features[:, 0, ...] = features[torch.arange(B), last_states_pos]

        late_diff_features = features - shifted_features

        return features, early_diff_features, late_diff_features

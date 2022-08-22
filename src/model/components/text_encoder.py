from functools import wraps

import torch
from torch import nn

from src.model.components.clip import clip

_OUTPUT_DIM = {
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


def dim_agnostic_encode(func):
    """Encode text with model from shape ([?], L) to ([?], d)"""

    @wraps(func)
    def wrapper(self, text):
        flat_dim, keep_dim = list(text.shape[:-1]), list(text.shape[-1:])
        pre_encoder_dim = [-1] + keep_dim
        post_encoder_dim = flat_dim + [-1]

        out = text.view(pre_encoder_dim)
        out = func(self, out)
        out = out.view(post_encoder_dim)

        return out

    return wrapper


class TextCLIP(nn.Module):
    def __init__(
        self,
        name: str = "ViT-L/14",
        pretrained: bool = True,
        fixed: bool = True,
        convert_fp16: bool = False,
    ):
        super().__init__()
        self.name = name
        model, _ = clip.load(
            name, pretrained=pretrained, convert_fp16=convert_fp16, device="cpu"
        )
        model = model.train()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

        self.context_length = model.context_length
        self.dtype = model.dtype
        self.output_dim = _OUTPUT_DIM[name]

        if not pretrained:
            self.fixed = False
        else:
            self.fixed = fixed

        for param in self.parameters():
            param.requires_grad = not self.fixed

    @dim_agnostic_encode
    def forward(self, label_ids):
        if self.fixed:
            self.eval()

        B, L = label_ids.size()
        assert L <= self.context_length

        if L != self.context_length:
            label_ids = torch.cat(
                (
                    label_ids,
                    torch.zeros(
                        B, self.context_length - L, dtype=label_ids.dtype
                    ),
                ),
                dim=1,
            )

        x = self.token_embedding(label_ids).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), label_ids.argmax(dim=-1)]
            @ self.text_projection
        )

        x = x / x.norm(dim=-1, keepdim=True)

        return x

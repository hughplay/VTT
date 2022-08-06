import torch
from torch import nn

from src.model.components.clip import clip


class ImageCLIP(nn.Module):
    def __init__(
        self,
        name: str = "ViT-B/32",
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
        self.encoder = model.visual

        if not pretrained:
            self.fixed = False
        else:
            self.fixed = fixed

        for param in self.encoder.parameters():
            param.requires_grad = not self.fixed

    def forward(self, images):
        if self.fixed:
            self.eval()
        out = self.encoder(images)
        return out


class TextCLIP(nn.Module):
    def __init__(
        self,
        name: str = "ViT-B/32",
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
        self.position_embedding = model.position_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.dtype = model.dtype

        if not pretrained:
            self.fixed = False
        else:
            self.fixed = fixed

        for param in self.encoder.parameters():
            param.requires_grad = not self.fixed

    def forward(self, text):
        if self.fixed:
            self.eval()

        x = self.token_embedding(text).type(
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
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.text_projection
        )

        return x

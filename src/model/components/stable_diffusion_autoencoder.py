# https://huggingface.co/blog/stable_diffusion
import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torch import nn


class StableDiffusionAutoEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        fixed: bool = True,
        size_in: float = 224,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        self.fixed = fixed
        if self.fixed:
            for param in self.vae.parameters():
                param.requires_grad = False

        self.size_in = size_in
        self.size_out = self.size_in // 8
        self.channel = 4

        self.c_embed = self.size_out**2 * self.channel

    def encode(self, images: torch.Tensor):
        if self.fixed:
            self.eval()
        latent = self.vae.encode(images).latent_dist.mode()
        latent = rearrange(
            latent,
            "b c h w -> b (c h w)",
            c=self.channel,
            h=self.size_out,
            w=self.size_out,
        )
        return latent

    def decode(self, latent: torch.Tensor):
        if self.fixed:
            self.eval()
        latent = rearrange(
            latent,
            "b (c h w) -> b c h w",
            c=self.channel,
            h=self.size_out,
            w=self.size_out,
        )
        return self.vae.decode(latent).sample

# https://huggingface.co/blog/stable_diffusion
import torch
from diffusers import AutoencoderKL
from torch import nn


class StableDiffusionAutoEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )

    def encode(self, images: torch.Tensor):
        z = self.vae.encode(images).latent_dist.mode()
        return z

    def decode(self, z: torch.Tensor):
        return self.vae.decode(z).sample

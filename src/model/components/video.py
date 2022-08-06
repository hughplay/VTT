from typing import Optional

import torch
from einops import rearrange
from torch import nn

from src.model.components.clipwrapper import ImageCLIP


class VideoEncoder(nn.Module):

    AGG_METHODS = ["mean", "max", "conv", "attn"]

    def __init__(
        self,
        backbone: nn.Module = None,
        max_batch_size: Optional[int] = None,
        agg_method: str = "mean",
        c_time: int = 3,
        k_conv: int = 3,
    ):
        super().__init__()
        if backbone is None:
            backbone = ImageCLIP()
        self.backbone = backbone
        self.max_batch_size = max_batch_size
        self.agg_method = agg_method

        assert agg_method in self.AGG_METHODS, f"{agg_method} is not supported"
        if self.agg_method == "conv":
            stride = 1
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=c_time,
                    out_channels=1,
                    kernel_size=k_conv,
                    stride=stride,
                    padding=(k_conv - stride) // 2,
                ),
                nn.ReLU(),
            )

        if self.agg_method == "attn":
            stride = 1
            self.attn = nn.Sequential(
                nn.Conv1d(
                    in_channels=c_time,
                    out_channels=c_time,
                    kernel_size=k_conv,
                    stride=stride,
                    padding=(k_conv - stride) // 2,
                ),
                nn.Softmax(dim=1),
            )
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=c_time,
                    out_channels=c_time,
                    kernel_size=k_conv,
                    stride=stride,
                    padding=(k_conv - stride) // 2,
                ),
            )

    def _agg(self, videos):
        if self.agg_method == "mean":
            agg_feats = videos.mean(dim=1)
        elif self.agg_method == "max":
            agg_feats, _ = videos.max(dim=1)
        elif self.agg_method == "conv":
            agg_feats = self.conv(videos)
            agg_feats = torch.squeeze(agg_feats, dim=1)
        elif self.agg_method == "attn":
            weight = self.attn(videos)
            value = self.conv(videos)
            agg_feats = torch.sum(weight * value, dim=1)
        else:
            raise NotImplementedError
        # video wise norm
        agg_feats = agg_feats / agg_feats.norm(dim=-1, keepdim=True)
        return agg_feats

    def forward(self, videos):
        B, T, _, _, _ = videos.shape
        device = next(self.backbone.parameters()).device

        frames = rearrange(videos, "b t c h w -> (b t) c h w")
        if self.max_batch_size is None:
            frames = self.backbone(frames.to(device))
        else:
            batch_frames = []
            for i in range(0, B * T, self.max_batch_size):
                batch = frames[i : i + self.max_batch_size]
                batch_frames.append(self.backbone(batch.to(device)))
            frames = torch.cat(batch_frames, dim=0)

        # norm through frame wise
        frames = frames / frames.norm(dim=-1, keepdim=True)
        videos = rearrange(frames, "(b t) c -> b t c", b=B, t=T)
        # B T C -> B C
        videos = self._agg(videos)

        return videos

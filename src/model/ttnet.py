import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from x_transformers import Decoder, Encoder

from .base import VideoEncoder
from .clip import CLIPImageEncoder

logger = logging.getLogger(__name__)


def exists(val):
    return val is not None


def max_negative_value(tensor):
    return -torch.finfo(tensor.dtype).max


def build_state_encoder(*args, backbone=None, **kwargs):
    return backbone


def build_clip_encoder(*args, backbone=None, **kwargs):
    return VideoEncoder(backbone, *args, **kwargs)


class StepFormer(nn.Module):
    """Find the correct clips from candidates with a feasible order."""

    def __init__(
        self, cfg: Dict[str, Any], max_clips: int = 7, max_gen_clips: int = 10
    ):
        super().__init__()

        self.cfg = cfg
        self.IGNORE_INDEX = int(cfg.IGNORE_INDEX)
        self.tau = cfg.tau
        self.max_clips = max_clips + 1  # for training, +1 for end token
        self.max_gen_clips = (
            max_gen_clips + 1
        )  # for inference, +1 for end token

        self.order = cfg.order
        assert self.order in ["left2right", "right2left"]

        backbone = CLIPImageEncoder(**cfg.backbone)
        self.state_encoder = build_state_encoder(
            backbone=backbone, **cfg.state_encoder
        )
        self.clip_encoder = build_clip_encoder(
            backbone=backbone, **cfg.clip_encoder
        )

        self.context_encoder = Encoder(**cfg.context_encoder)
        self.context_norm = nn.LayerNorm(cfg.dim)

        self.decoder = Decoder(**cfg.decoder)
        self.decoder_norm = nn.LayerNorm(cfg.dim)

    def _append_end_token(self, candidates, end_token=None, targets=None):
        # append the end token to candidates, B * (N + 1) * d
        candidates = torch.cat([candidates, end_token.unsqueeze(1)], dim=1)

        if exists(targets):
            # `the length of candidates` is the end mark between steps and
            # distractors, no more operations needed for targets.
            return candidates, targets

        return candidates

    def _get_step_mask(self, targets):
        mask = torch.ones_like(targets).bool()
        pos_end = targets.argmax(dim=1)
        for i, pos in enumerate(pos_end):
            mask[i, pos + 1 :] = False
        return mask

    def _encode_inputs(self, init, fin, candidates=None):
        # prepare tokens (state, context, or clip)
        # B * 3 * H * W -> B * 1 * d
        init = self.state_encoder(init)
        # B * 3 * H * W -> B * 1 * d
        fin = self.state_encoder(fin)
        # B * N * T * 3 * H * W -> B * N * d
        if candidates is None:
            B = init.shape[0]
            candidates = self.full_candidates.expand(B, -1, -1)
        else:
            candidates = self._encode_candidates(candidates)
        # context, B * 2 * d -> B * 2 * d
        context = self.context_norm(
            self.context_encoder(torch.stack([init, fin], dim=1))
            # self.context_encoder(torch.stack([init, fin, fin - init], dim=1))
        )

        return init, fin, candidates, context

    def _encode_candidates(self, candidates):
        # B * N * T * 3 * H * W -> B * N * d
        B, N = candidates.shape[:2]
        candidates = rearrange(
            self.clip_encoder(
                rearrange(candidates, "b n t c h w -> (b n) t c h w")
            ),
            "(b n) d -> b n d",
            b=B,
            n=N,
        )
        return candidates

    def _de_permute(self, vectors, targets):
        """de-permute the vectors according to targets."""
        # TODO check
        order = targets[..., None].expand_as(vectors)
        sorted_vectors = torch.gather(vectors, 1, order)
        return sorted_vectors

    def _reverse(self, targets, inplace=False):
        if not inplace:
            targets = targets.clone()
        pos_end = targets.argmax(dim=1)
        for i, pos in enumerate(pos_end):
            targets[i, :pos] = targets[i, :pos].flip(dims=(0,))
        return targets

    def load_candidates(self, candidates):
        self.full_candidates = self._encode_candidates(candidates.unsqueeze(0))

    def forward(
        self,
        init: torch.Tensor,
        fin: torch.Tensor,
        candidates: torch.Tensor = None,
        targets: torch.Tensor = None,
    ):
        """forward function of reasoner
        Note:
            B : batch size
            d : hidden dimension
            H : height, W : width, T : frames
            N : the number of candidates
        Parameters
        ----------
        init : torch.Tensor, B * 3 * H * W
            initial state
        fin : torch.Tensor, B * 3 * H * W
            final state
        candidates : torch.Tensor, B * N * T * 3 * H * W
            candidate clips
        targets : torch.Tensor, optional, B * (N + 1)
            indices of ground truth clips in the correct order
        """

        # inference mode
        if not exists(targets):
            return self.inference(init, fin, candidates)

        init, fin, candidates, context = self._encode_inputs(
            init, fin, candidates
        )

        if self.order == "left2right":
            start_token, end_token = init, fin
        elif self.order == "right2left":
            start_token, end_token = fin, init
            targets = self._reverse(targets)

        # fin as end token
        # candidates: B * (N + 1) * d, targets:  B * (N + 1)
        candidates, targets = self._append_end_token(
            candidates, end_token=end_token, targets=targets
        )

        # gather GT vectors
        # v_targets: B * (N + 1) * d
        v_targets = self._de_permute(candidates, targets)[
            :, : self.max_clips, :
        ]
        targets = targets[:, : self.max_clips]

        # inputs, v_pred: B * (N + 1) * d
        inputs = torch.cat([start_token.unsqueeze(1), v_targets], dim=1)[
            :, :-1, :
        ]
        mask = self._get_step_mask(targets)
        v_pred = self.decoder_norm(
            self.decoder(inputs, mask=mask, context=context)
        )

        # Note: in nn.functional.cross_entropy, the dimension of inputs and
        # targets are `N * C * ...` and `N * ...` respectively.
        # In our cases, candidates are equivalent to classes, therefore, the
        # dimension of logits is: B * n_candidate * n_step.
        logits = (
            torch.einsum("b i h, b j h -> b i j", candidates, v_pred) / self.tau
        )
        masked_targets = targets * mask + self.IGNORE_INDEX * ~mask
        loss = F.cross_entropy(
            logits, masked_targets, ignore_index=self.IGNORE_INDEX
        )
        steps = torch.argmax(logits, dim=1)

        if self.order == "right2left":
            steps = self._reverse(steps)

        return {"loss": loss, "steps": steps}

    @torch.no_grad()
    def inference(
        self,
        init: torch.Tensor,
        fin: torch.Tensor,
        candidates: torch.Tensor = None,
    ):
        self.eval()

        init, fin, candidates, context = self._encode_inputs(
            init, fin, candidates
        )

        if self.order == "left2right":
            start_token, end_token = init, fin
        elif self.order == "right2left":
            start_token, end_token = fin, init

        # fin as end token
        # candidates, targets: B * (N + 1) * d
        candidates = self._append_end_token(candidates, end_token=end_token)

        B, N, _ = candidates.shape
        # B * 1 * d
        inputs = start_token.unsqueeze(1)
        preds = []
        mask = torch.zeros(B, N).bool()
        for _ in range(self.max_gen_clips):
            v_pred = self.decoder_norm(self.decoder(inputs, context=context))
            logits = torch.einsum(
                "b i h, b j h -> b i j", candidates, v_pred[:, -1:, :]
            )
            logits[mask] = max_negative_value(logits)
            pred = torch.argmax(logits, dim=1).squeeze()
            preds.append(pred)

            # extend inputs
            inputs = torch.cat(
                [inputs, candidates[torch.arange(B), pred].unsqueeze(1)], dim=1
            )
            # mask out selected candidates
            mask[torch.arange(B), pred] = True

        steps = torch.stack(preds, dim=1)

        if self.order == "right2left":
            steps = self._reverse(steps)

        return {"steps": steps}

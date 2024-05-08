import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

try:
    from torchvision.transforms.functional import InterpolationMode
except Exception:
    from PIL import Image as InterpolationMode


logger = logging.getLogger(__name__)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_INVERSE_MEAN = (-mean / std for mean, std in zip(CLIP_MEAN, CLIP_STD))
CLIP_INVERSE_STD = (1.0 / std for std in CLIP_STD)
IMAGENET_INVERSE_MEAN = (
    -mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)
)
IMAGENET_INVERSE_STD = (1.0 / std for std in IMAGENET_STD)


def forward_transform(n_px=224, mode="clip"):
    """forward image transform
    Parameters
    ----------
    n_px :
        resolution of input
    Returns
    -------
    torchvison.transforms.Compose
        torchvision transform
    """
    transforms = [
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
    ]

    if mode == "clip":
        transforms.append(Normalize(CLIP_MEAN, CLIP_STD))
    elif mode == "imagenet":
        transforms.append(Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif mode == "stable_diffusion":
        transforms.append(Lambda(lambd=lambda x: x * 2 - 1))
    else:
        raise ValueError(f"mode {mode} is not supported")

    return Compose(transforms)


def inverse_transform(mode="clip"):
    """inverse of clip image transform
    Returns
    -------
    torchvision.transforms.Compose
        torchvision transform
    """
    transforms = [
        ToPILImage(),
    ]
    if mode == "clip":
        transforms.insert(0, Normalize(CLIP_INVERSE_MEAN, CLIP_INVERSE_STD))
    elif mode == "imagenet":
        transforms.insert(
            0, Normalize(IMAGENET_INVERSE_MEAN, IMAGENET_INVERSE_STD)
        )
    elif mode == "stable_diffusion":
        transforms.insert(0, Lambda(lambd=lambda x: (x + 1) / 2))
        transforms.insert(1, Lambda(lambd=lambda x: x.clamp(0, 1)))
    else:
        raise ValueError(f"mode {mode} is not supported")

    return Compose(transforms)


class ConsistentTransform:
    """Transform multiple images with same settings
    Usage:
        transform = ConsistentTransform(
            n_px=224, resize=False, random_crop=False, random_flip=True
        )
        transform.step()
        image_a1 = transform(image_a1)
        image_a2 = transform(image_a2)
        transform.step()
        image_b1 = transform(image_b1)
        image_b2 = transform(image_b2)
    """

    def __init__(
        self,
        n_px=224,
        resize=False,
        random_crop=False,
        random_hflip=False,
        normalize=True,
        transform_mode="clip",
    ):

        self.n_px = n_px
        self.resize = resize
        self.random_crop = random_crop
        self.random_hflip = random_hflip

        if transform_mode == "clip":
            mean, std = CLIP_MEAN, CLIP_STD
            inverse_mean, inverse_std = CLIP_INVERSE_MEAN, CLIP_INVERSE_STD
        elif transform_mode == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
            inverse_mean, inverse_std = (
                IMAGENET_INVERSE_MEAN,
                IMAGENET_INVERSE_STD,
            )
        else:
            raise ValueError(f"mode {transform_mode} is not supported")

        self._to_tensor = (
            Compose(
                [
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(mean, std),
                ]
            )
            if normalize
            else Compose(
                [
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                ]
            )
        )
        self._to_pil = Compose(
            [
                Normalize(inverse_mean, inverse_std),
                ToPILImage(),
            ]
        )

        self._state_never_set = True

    def _resize(self, image):
        if self.resize:
            image = TF.resize(
                image, self.n_px, interpolation=InterpolationMode.BICUBIC
            )
        return image

    def _get_random_crop_params(self, img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = TF._get_image_size(img)
        th, tw = self.n_px, self.n_px

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format(
                    (th, tw), (h, w)
                )
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def _crop(self, image):
        if self.random_crop:
            image = TF.crop(image, *self._crop_state)
        else:
            image = TF.center_crop(image, self.n_px)
        return image

    def _hflip(self, image):
        if self.random_hflip and self._hflip_state < 0.5:
            image = TF.hflip(image)
        return image

    def _change_state(self, image=None):
        self._state_never_set = False
        self._hflip_state = random.random()
        if self.random_crop:
            assert image is not None
            self._crop_state = self._get_random_crop_params(image)

    def transform(self, image, change_state=False):
        if self.resize:
            image = self._resize(image)
        if change_state or self._state_never_set:
            self._change_state(image)
        image = self._crop(image)
        image = self._hflip(image)
        image = self._to_tensor(image)
        return image

    def to_pil(self, image):
        return self._to_pil(image)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class VideoFrameReader:
    def __init__(
        self,
        n_segment: int = 3,
        frames_per_segment: int = 1,
        list2tensor: bool = False,
        transform=None,
    ):
        self.n_segment = n_segment
        self.frames_per_segment = frames_per_segment
        self.list2tensor = list2tensor

        if transform is None:
            self.transform = forward_transform()
        else:
            self.transform = transform

    """VideoSegmentReader from jpg frames"""

    def _get_frame_list(self, video_path):
        """refer to: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/
        blob/212bc039f0d21c633d2728dfd7459f9efa7ac2e2/video_dataset.py#L143 For
        each segment, chooses an index from where frames are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        frames = list(sorted(Path(video_path).expanduser().glob("*.jpg")))
        n_frames = len(frames)

        segment_duration = (
            n_frames - self.frames_per_segment + 1
        ) // self.n_segment
        if segment_duration > 0:
            frame_list_index = np.multiply(
                list(range(self.n_segment)), segment_duration
            ) + np.random.randint(segment_duration, size=self.n_segment)

        # edge cases for when a video has approximately less than
        # (num_frames*frames_per_segment) frames. random sampling in that case,
        # which will lead to repeated frames.
        else:
            frame_list_index = np.sort(
                np.random.randint(n_frames, size=self.n_segment)
            )
        frame_list = [frames[index] for index in frame_list_index]

        return frame_list

    def _get_frame(self, path):
        tensor = self.transform(Image.open(path).convert("RGB"))
        return tensor

    def sample(self, video_path: str):
        max_try = 5
        for _ in range(max_try):
            try:
                tensors = [
                    self._get_frame(path)
                    for path in self._get_frame_list(video_path)
                ]
                break
            except Exception as e:
                logger.error(str(e))
                time.sleep(1)

        if self.list2tensor:
            tensors = torch.stack(tensors, axis=0)
        return tensors

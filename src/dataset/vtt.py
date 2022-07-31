import logging
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

import src.utils.datatool as dtool
from src.dataset.text import SimpleTokenizer
from src.dataset.vision import ConsistentTransform, VideoFrameReader

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VTTDataset(Dataset):

    CHANNEL = 3

    def __init__(
        self,
        data_root: str = "/data/vtt",
        meta_path: str = "meta/vtt.jsonl",
        state_root: str = "states",
        frame_root: str = "frames",
        split: str = "train",
        load_trans_frames: bool = False,
        n_segment: int = 3,
        frames_per_segment: int = 1,
        transform_cfg: Dict[str, Any] = {},
        max_n_transformation: int = 12,
        max_n_words: int = 24,
    ):
        self.data_root = Path(data_root).expanduser()
        self.data = dtool.JSONLList(
            self.data_root / meta_path, lambda x: x["split"] == split
        )
        self.state_root = self.data_root / state_root
        self.frame_root = self.data_root / frame_root
        self.load_frames = load_trans_frames

        self.tokenizer = SimpleTokenizer()
        self.transform = ConsistentTransform(**transform_cfg)
        if self.load_frames:
            self.n_segment = n_segment
            self.frames_per_segment = frames_per_segment
            self.video_reader = VideoFrameReader(
                n_segment=n_segment,
                frames_per_segment=frames_per_segment,
                list2tensor=True,
                transform=self.transform,
            )

        self.max_n_transformation = max_n_transformation
        self.max_n_states = max_n_transformation + 1
        self.max_n_words = max_n_words

    def __len__(self):
        return len(self.data)

    def _read_labels(self, sample):
        """n_trans * n_words"""
        ids = torch.empty(
            self.max_n_transformation, self.max_n_words, dtype=torch.int64
        )
        ids.fill_(self.tokenizer.pad_idx)
        mask = torch.zeros(
            self.max_n_transformation, self.max_n_words, dtype=torch.bool
        )

        for i, step in enumerate(sample["annotation"]):
            words = torch.tensor(
                [self.tokenizer.start_idx]
                + self.tokenizer.encode(step["label"])
                + [self.tokenizer.end_idx]
            )
            ids[i, : len(words)] = words
            mask[i, : len(words)] = True

        return ids, mask

    def _read_states(self, sample):
        """n_states * C * H * W"""
        n_states = len(sample["annotation"]) + 1
        states_path_list = [
            self.state_root / f"{sample['id']}_{n_states}_{i}.jpg"
            for i in range(n_states)
        ]
        states = torch.zeros(
            self.max_n_states,
            self.CHANNEL,
            self.transform.n_px,
            self.transform.n_px,
            dtype=torch.float,
        )
        mask = torch.zeros(self.max_n_states, dtype=torch.bool)

        # the state of ConsistTransform is changed here only once for each sample
        # so that all states, frames are augmented (resize, crop, flip) with the
        # same arguments
        for (i, state_path) in enumerate(states_path_list):
            states[i] = self.transform(
                Image.open(str(state_path)), change_state=(i == 0)
            )
        mask[: len(states_path_list)] = True
        return states, mask

    def _read_trans_frames(self, sample):
        """n_trans * T * C * H * W, T: frames sampled per video clip"""
        n_steps = len(sample["annotation"])
        clips_root_list = [
            self.frame_root / f"{sample['id']}_{n_steps}_{i}"
            for i in range(n_steps)
        ]
        clips = torch.zeros(
            self.max_n_transformation,
            self.n_segment * self.frames_per_segment,
            self.CHANNEL,
            self.transform.n_px,
            self.transform.n_px,
            dtype=torch.float,
        )
        mask = torch.zeros(self.max_n_transformation, dtype=torch.bool)

        for i, clip_root in enumerate(clips_root_list):
            clips[i] = self.video_reader.sample(clip_root)
        mask[: len(clips_root_list)] = True
        return clips, mask

    def __getitem__(self, index):
        meta = self.data[index]
        label_ids, label_mask = self._read_labels(meta)
        states, states_mask = self._read_states(meta)
        res = {
            "label_ids": label_ids,
            "label_mask": label_mask,
            "states": states,
            "states_mask": states_mask,
        }
        if self.load_frames:
            trans, trans_mask = self._read_trans_frames(meta)
            res["trans"] = trans
            res["trans_mask"] = trans_mask
        return res


class VTTDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        dataset_cfg: Dict[str, Any] = None,
        transform_cfg: Dict[str, Any] = None,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_cfg = dataset_cfg
        self.transform_cfg = transform_cfg

    def _dataloader(self, split):
        if split in ["train"]:
            dataset_cfg = self.dataset_cfg["train"]
            transform_cfg = self.transform_cfg["train"]
        else:
            dataset_cfg = self.dataset_cfg["eval"]
            transform_cfg = self.transform_cfg["eval"]

        dataset = VTTDataset(
            **dataset_cfg, split=split, transform_cfg=transform_cfg
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=(split == "train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

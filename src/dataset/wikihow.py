import logging
from pathlib import Path

import torch
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

import src.utils.datatool as dtool
from src.dataset.text import SimpleTokenizer

logger = logging.getLogger(__name__)


class WikiHowStep(Dataset):
    def __init__(
        self,
        split: str = "train",
        data_root: str = "/data/vtt/wikihow",
        embedding_model: str = "ViT-L/14",
        max_words: int = 77,
    ):
        self.split = split
        self.data_root = Path(data_root).expanduser()
        self.text_path = self.data_root / f"wikihow_{split}.txt"
        self.samples = dtool.read_lines(self.text_path, strip=True)
        self.embedding = LmdbEmbeddingsReader(
            str(
                self.data_root
                / f"wikihow_{embedding_model.replace('/', '_')}.lmdb"
            )
        )
        self.tokenizer = SimpleTokenizer(max_words=max_words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.samples[index]
        label_ids, label_mask = self.tokenizer.tokenize(
            [text], return_mask=True
        )

        embedding = self.embedding.get_word_vector(f"{self.split}_{index}")

        return {
            "label_ids": label_ids.squeeze(),
            "label_mask": label_mask.squeeze(),
            "embedding": embedding,
            "index": index,
        }


class WikiHowDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "/data/vtt/wikihow",
        embedding_model: str = "ViT-L/14",
        max_words: int = 77,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root).expanduser()
        self.embedding_model = embedding_model
        self.max_words = max_words
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _dataloader(self, split: str):

        dataset = WikiHowStep(
            split=split,
            data_root=self.data_root,
            embedding_model=self.embedding_model,
            max_words=self.max_words,
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

import logging

import pytest
import torch

from src.dataset.wikihow import WikiHowDataModule

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model", ["ViT-B/32", "ViT-B/16", "ViT-L/14"])
def test_wikihow_datamodule(model):
    batch_size = 4

    datamodule = WikiHowDataModule(batch_size=batch_size, embedding_model=model)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader
    assert val_loader
    assert test_loader

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    batch = next(iter(train_loader))

    assert len(batch["label_ids"]) == batch_size
    assert len(batch["label_mask"]) == batch_size
    assert len(batch["embedding"]) == batch_size

    assert batch["label_ids"].dtype == torch.long
    assert batch["label_mask"].dtype == torch.bool
    assert batch["embedding"].dtype == torch.float

    assert len(batch["label_ids"].shape) == 2
    assert len(batch["label_mask"].shape) == 2
    assert len(batch["embedding"].shape) == 2

    assert batch["label_ids"].shape[1] == 77
    assert batch["label_mask"].shape[1] == 77
    if model == "ViT-L/14":
        assert batch["embedding"].shape[1] == 768
    else:
        assert batch["embedding"].shape[1] == 512

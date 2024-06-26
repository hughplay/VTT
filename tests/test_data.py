import logging

import dotenv
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.dataset.text import SimpleTokenizer
from src.dataset.vtt import CATEGORIES, TOPICS
from src.utils.exptool import register_omegaconf_resolver

logger = logging.getLogger(__name__)

register_omegaconf_resolver()
dotenv.load_dotenv("../.env", override=True)


@pytest.mark.parametrize("prefix_start", [True, False])
def test_vtt_datamodule(prefix_start):
    batch_size = 4
    load_frames = False

    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"dataset.batch_size={batch_size}",
                f"dataset.dataset_cfg.train.load_trans_frames={load_frames}",
                f"dataset.dataset_cfg.train.prefix_start={prefix_start}",
            ],
        )
    datamodule = instantiate(cfg.dataset)

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

    assert len(batch["states"]) == batch_size
    assert len(batch["states_mask"]) == batch_size
    assert len(batch["label_ids"]) == batch_size
    assert len(batch["label_mask"]) == batch_size
    assert len(batch["category"]) == batch_size
    assert len(batch["category"]) == batch_size
    if load_frames:
        assert len(batch["trans"]) == batch_size
        assert len(batch["trans_mask"]) == batch_size

    assert batch["states"].dtype == torch.float
    assert batch["states_mask"].dtype == torch.bool
    assert batch["label_ids"].dtype == torch.int64
    assert batch["label_mask"].dtype == torch.bool
    assert batch["category"].dtype == torch.int64
    assert batch["topic"].dtype == torch.int64
    if load_frames:
        assert batch["trans"].dtype == torch.float
        assert batch["trans_mask"].dtype == torch.bool

    assert len(batch["states"].shape) == 5
    assert len(batch["states_mask"].shape) == 2
    assert len(batch["label_ids"].shape) == 3
    assert len(batch["label_mask"].shape) == 3
    assert len(batch["category"].shape) == 1
    assert len(batch["topic"].shape) == 1
    if load_frames:
        assert len(batch["trans"].shape) == 6
        assert len(batch["trans_mask"].shape) == 2

    tokenizer = SimpleTokenizer()
    B, N, L = batch["label_ids"].size()
    for i in range(B):
        for j in range(N):
            if torch.any(batch["label_mask"][i, j]):
                if prefix_start:
                    assert batch["label_ids"][i, j][0] == tokenizer.start_idx
                else:
                    assert batch["label_ids"][i, j][0] != tokenizer.start_idx
                end_pos = batch["label_mask"][i, j].sum() - 1
                assert batch["label_ids"][i, j][end_pos] == tokenizer.end_idx

    assert torch.all(batch["category"] < len(CATEGORIES))
    assert torch.all(batch["topic"] < len(TOPICS))

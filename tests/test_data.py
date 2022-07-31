from pathlib import Path

import dotenv
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.utils.exptool import register_omegaconf_resolver

register_omegaconf_resolver()
dotenv.load_dotenv(override=True)


@pytest.mark.parametrize("batch_size", [1])
def test_mnist_datamodule(batch_size):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="train", overrides=[f"dataset.batch_size={batch_size}"]
        )
    datamodule = instantiate(cfg.dataset)

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))

    assert len(batch["label_ids"]) == batch_size
    assert len(batch["label_mask"]) == batch_size
    assert len(batch["states"]) == batch_size
    assert len(batch["states_mask"]) == batch_size
    assert len(batch["trans"]) == batch_size
    assert len(batch["trans_mask"]) == batch_size

    assert batch["label_ids"].dtype == torch.int64
    assert batch["label_mask"].dtype == torch.bool
    assert batch["states"].dtype == torch.float
    assert batch["states_mask"].dtype == torch.bool
    assert batch["trans"].dtype == torch.float
    assert batch["trans_mask"].dtype == torch.bool

    assert len(batch["label_ids"].shape) == 3
    assert len(batch["label_mask"].shape) == 3
    assert len(batch["states"].shape) == 5
    assert len(batch["states_mask"].shape) == 2
    assert len(batch["trans"].shape) == 6
    assert len(batch["trans_mask"].shape) == 2

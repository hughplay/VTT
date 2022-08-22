import logging
from typing import Any, Dict, List

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info

logger = logging.getLogger(__name__)


class TellingLitModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.model = instantiate(self.hparams.model)
        self.criterion = instantiate(self.hparams.criterion)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optim, self.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.hparams.scheduler), optimizer
        )
        # torch's schedulers are epoch-based, but transformers' are step-based
        interval = (
            "step"
            if self.hparams.scheduler._target_.startswith("transformers")
            else "epoch"
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            if self.global_rank == 0:
                logger.info("Computing number of training steps...")
                num_training_steps = [self.trainer.estimated_stepping_batches]
            else:
                num_training_steps = [0]
            torch.distributed.broadcast_object_list(
                num_training_steps,
                0,
                group=torch.distributed.group.WORLD,
            )
            scheduler_cfg["num_training_steps"] = num_training_steps[0]

            if self.global_rank == 0:
                logger.info(
                    f"Training steps: {scheduler_cfg['num_training_steps']}"
                )
        return scheduler_cfg

    def on_train_start(self):
        self.criterion.reset()

    def step(self, batch: Any, eval=False):
        states = batch["states"]
        states_mask = batch["states_mask"]
        label_ids = batch["label_ids"]
        label_mask = batch["label_mask"]
        inputs = {
            "states": states,
            "states_mask": states_mask,
            "label_ids": label_ids,
            "label_mask": label_mask,
        }

        outputs = self.model(**inputs)
        outputs.update(
            {
                "label_ids": label_ids,
                "label_mask": label_mask,
                "category": batch["category"],
                "topic": batch["topic"],
            }
        )

        # Compute loss and metrics, input keys and values are reserved
        outputs = self.criterion(outputs, eval=eval)
        return outputs

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, eval=False)

        self.log(
            "train/loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/ppl",
            self.criterion.perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return outputs["loss"]

    def on_validation_start(self) -> None:
        self.criterion.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, eval=True)

        self.log(
            "val/loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def validation_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute()
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
            )

    def on_test_start(self) -> None:
        self.criterion.reset()

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, eval=True)

        self.log(
            "test/loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def test_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute()
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=True,
            )

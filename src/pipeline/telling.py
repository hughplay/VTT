import logging
from typing import Any, Dict, List, Sequence, Union

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)


class TellingLitModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any] = None):
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

    def step(
        self,
        batch: Any,
        compute_loss: bool = True,
        update_eval: bool = True,
        exclude_eval_metrics: Union[str, Sequence[str]] = None,
        generate_from_scratch: bool = False,
    ):
        """The choice of one forward step:

        - computing loss
        - update evaluation metrics
            - exclude evaluation metrics
        - generate from scratch
        """
        states = batch["states"]
        states_mask = batch["states_mask"]
        label_ids = batch["label_ids"]
        label_mask = batch["label_mask"]

        if generate_from_scratch:
            inputs = {
                "states": states,
                "states_mask": states_mask,
            }
            compute_loss = False
        else:
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
                "index": batch["index"],
            }
        )

        # Compute loss and metrics, input keys and values are reserved
        outputs = self.criterion(
            outputs,
            compute_loss=compute_loss,
            update_eval=update_eval,
            exclude_eval_metrics=exclude_eval_metrics,
        )

        return outputs

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, compute_loss=True, update_eval=False)

        self.log(
            "train/loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/PPL",
            self.criterion.perplexity.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return outputs["loss"]

    def on_validation_start(self) -> None:
        self.criterion.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, update_eval=True, generate_from_scratch=True)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute(verbose=True)
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
        outputs = self.step(batch, update_eval=True, generate_from_scratch=True)
        return outputs

    def test_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=True,
            )
        self.criterion.save()

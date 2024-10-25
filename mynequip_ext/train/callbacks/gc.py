import torch
import lightning
from lightning.pytorch.callbacks import Callback

import gc
from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule


class GarbageCollector(Callback):

    def __init__(
        self,
        interval: str,
        frequency: int,
    ):
        assert interval in ["batch", "epoch"]
        assert frequency >= 1
        self.interval = interval
        self.frequency = frequency

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs: torch.Tensor,
        batch: AtomicDataDict.Type,
        batch_idx: int,
    ) -> None:
        """"""
        if self.interval == "batch":
            if trainer.global_step % self.frequency == 0:
                gc.collect()

    def on_train_epoch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        if self.interval == "epoch":
            if trainer.current_epoch % self.frequency == 0:
                gc.collect()
import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from hs_prediction.loss.loss import batch_loss, loss
from hs_prediction.models.location.network import EGNN
from hs_prediction.models.utils import get_optimizer, get_scheduler, unload


class LightningModel(pl.LightningModule):
    """PyTorch Lightning Model"""

    def __init__(
        self,
        config: DictConfig,
        network: EGNN,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler: Optional[str] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__()
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_scheduler = optimizer_scheduler
        self.optimizer_scheduler_kwargs = optimizer_scheduler_kwargs
        self.loss = loss
        self.train_outputs = []
        self.valid_outputs = []

    def predict(self, graph, save_path=None):
        """Predict step (forward pass)"""
        pred = self.network(graph, save_path)
        return pred

    def forward(self, batch, outputs=None):
        """Forward pass of the network"""
        try:
            certainty, water_prediction_pos, batch_water = self.network(batch)
            true_water_pos = batch["wat"].pos
            true_water_batch = batch["wat"].batch
            occupancy = batch["wat"]["occupancy"]
            loss = batch_loss(
                certainty,
                water_prediction_pos,
                batch_water,
                occupancy,
                true_water_pos,
                true_water_batch,
                self.loss,
                loss_config=self.config.loss,
            )
            print("loss: ", loss)
            if loss is not None and outputs is not None:
                outputs.append(
                    {
                        "loss": loss,
                        "pred_certainty": unload(certainty),
                        "water_prediction_pos": unload(water_prediction_pos),
                        "true_water_pos": unload(true_water_pos),
                    }
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory")
                return None
            else:
                raise e

        return loss

    @staticmethod
    def epoch_end_metrics(outputs, label: str, stride: int = 1):
        """Compute all metrics at the end of an epoch"""
        losses = [output["loss"] for output in outputs[::stride]]
        metrics = {
            f"{label}_loss": torch.tensor(losses).mean().item(),
        }
        return metrics

    def training_step(self, batch, batch_idx):
        """Training step (forward pass & loss)"""
        loss = self.forward(batch, self.train_outputs)
        return loss

    def on_train_epoch_end(self):
        """Training epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.train_outputs, "train", stride=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_outputs = []

    def validation_step(self, batch, batch_idx):
        """Validation step (forward pass & loss)"""
        loss = self.forward(batch, self.valid_outputs)
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.valid_outputs, "valid", stride=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.valid_outputs = []

    def configure_optimizers(self):
        """Configure optimizer and scheduler for training"""
        optimizer = get_optimizer(self, self.optimizer, **self.optimizer_kwargs)
        if self.optimizer_scheduler is None:
            return optimizer
        else:
            scheduler = get_scheduler(
                optimizer, self.optimizer_scheduler, **self.optimizer_scheduler_kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid_loss",
            }

    def load_checkpoint(self, checkpoint_path):
        assert os.path.exists(
            checkpoint_path
        ), f"resume_path ({checkpoint_path}) does not exist"
        self.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        print(f"Loaded checkpoint from {checkpoint_path}")


def create_model(config: DictConfig) -> LightningModel:
    """Create model from configuration"""
    return LightningModel(
        config=config, network=EGNN(**config.model), **config.training
    )

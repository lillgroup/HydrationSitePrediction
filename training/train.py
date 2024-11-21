# env: hydration_site2

import os
import resource

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from hs_prediction.data.dataset import create_dataloaders
from hs_prediction.models.location.model import create_model
from hs_prediction.models.thermodynamics.model_entropy import create_thermo_model
from hs_prediction.utils.auxiliary import InitialCheckpoint, set_seed


# @hydra.main(config_path="../config/", config_name="location_model", version_base="1.1")
@hydra.main(config_path="../config/", config_name="thermo_model", version_base="1.1")
def main(config):
    # Initialize W&B Run
    wandb.init(
        project=config.general.project_name,
        config=OmegaConf.to_container(config),
        reinit=True,
    )
    # Create train and validation dataloaders
    train_dataloader, valid_dataloader, _, _ = create_dataloaders(config)
    if config.general.project_name == "hs_location":
        model = create_model(config)
    elif config.general.project_name == "thermodynamics":
        model = create_thermo_model(config)
    if (
        config.training.resume_path is not None
        and config.general.load_checkpoint is True
    ):
        model.load_checkpoint(config.training.resume_path)
    initial_checkpoint_path = os.path.join(
        config.general.repo_dir, "outputs/initials/initial_checkpoint.ckpt"
    )
    trainer = Trainer(
        logger=WandbLogger(project=config.general.project_name),
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices=config.training.cuda_ids,
        precision=16,
        accumulate_grad_batches=config.training.acc_grad_batches,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath="outputs",
                monitor="valid_loss",
                mode="min",
                save_last=True,
            ),
            InitialCheckpoint(initial_ckp_path=initial_checkpoint_path),
        ],
        check_val_every_n_epoch=1,
    )
    # Train neural network
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()

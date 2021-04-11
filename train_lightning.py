import warnings

import src.configuration as C
import src.models as models
import src.utils as utils

from pathlib import Path

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



class LightningBirdcall(pl.LightningModule):

    def __init__(self, config,trn_idx, val_idx):
        super().__init__()
        self.config = config
        self.model = models.get_model(config).to(device)
        self.criterion = C.get_criterion(config).to(device)
        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        self.loaders = {
            phase: C.get_loader(df_, datadir, config, phase, event_level_labels)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
    def forward(self, img, meta):
        x = self.model(img, meta)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, meta, y = batch
        y_hat = self(x, meta)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        y_hat = self(x, meta)
        val_loss = self.criterion(y_hat, y)
        return val_loss
    def configure_optimizers(self):
        self.optimizer = C.get_optimizer(self.model, self.config)
        self.scheduler = C.get_scheduler(self.optimizer, self.config)
        return [self.optimizer], [self.scheduler]
    def train_dataloader(self):
          return self.loaders['train']
    def val_dataloader(self):
          return self.loaders['valid']

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")

    utils.set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)
    early_stop_callback = EarlyStopping(
      monitor='val_accuracy',
      min_delta=0.00,
      patience=3,
      verbose=False,
      mode='max'
    )
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    

    if config["data"].get("event_level_labels") is not None:
        event_level_labels = C.get_event_level_labels(config)
    else:
        event_level_labels = None

    for i, (trn_idx, val_idx) in enumerate(
            splitter.split(df, y=df["primary_label"])):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)
        checkpoint = ModelCheckpoint(f'./fold-{i}')
        lightning_model = LightningBirdcall(config, trn_idx, val_idx)
        trainer = pl.Trainer(gpus=1,logger=logger, \
            callbacks=[early_stop_callback, checkpoint], \
            auto_select_gpus=True, max_epochs=global_params["num_epochs"], accumulate_grad_batches=1 ,\
            limit_val_batches=0.4, gradient_clip_val=1)
        trainer.fit(lightning_model)
        
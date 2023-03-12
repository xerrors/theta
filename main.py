from utils.args import setup_parser
from models.theta import Theta
from config import Config
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.plugins import DDPPlugin

import utils
from data.data_module import DataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')


def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def main(func_mode=False, **kwargs):
    args = setup_parser(func_mode, **kwargs)
    config = Config(args, **kwargs)  # configure logging, save config, gpu etc.
    seed_every_thing(config.seed)

    data = DataModule(config)  # The data
    theta = Theta(config, data)  # The model

    # Trainer callbacks
    early_callback = pl.callbacks.EarlyStopping(
        monitor="val_f1", mode="max", patience=10, check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_f1", mode="max",
        filename='f1={val_f1:.4f}-epoch={epoch}',
        dirpath=os.path.join(config.output_dir, "checkpoints"),
        auto_insert_metric_name=False,
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint] if not config.fast_dev_run else []

    # gpu_count = torch.cuda.device_count()

    # Configure Trainer
    trainer = pl.Trainer.from_argparse_args(
        config.args,
        accelerator='gpu', devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        auto_lr_find=config.auto_lr,
        default_root_dir=config.output_dir,
        logger=configure_logger(config),
    )

    if config.auto_lr:
        print(utils.yellow("Auto LR Finding..."))
        trainer.tune(theta, datamodule=data)  # with auto_lr_find=True

    trainer.fit(theta, datamodule=data)
    config.save_best_model_path(model_checkpoint.best_model_path)
    config.save_config()

    trainer.test(theta, datamodule=data)

    wandb.finish()
    return theta.best_f1, theta.test_f1


def configure_logger(config):
    """ TensorBoardLogger (offline) or WandbLogger (Online) """

    if config.wandb:
        logger = pl.loggers.WandbLogger(project="theta", name=config.tag, save_dir=config.output_dir)
        logger.log_hyperparams(config)
    else:
        logger = pl.loggers.TensorBoardLogger(
            name=config.tag,
            save_dir=os.path.join(config.output_dir, "tensorboard"))

    return logger


if __name__ == "__main__":
    print(main())

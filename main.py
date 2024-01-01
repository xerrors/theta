from utils.args import setup_parser
from models.theta import Theta
from config import Config
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from data.data_module import DataModule
from xerrors import cprint as cp

import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    early_callback = pl.callbacks.EarlyStopping( # type: ignore
        monitor="val_f1", mode="max", patience=30, check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint( # type: ignore
        monitor="val_f1", mode="max", save_last=True,
        filename='f1={val_f1:.4f}-epoch={epoch}',
        dirpath=os.path.join(config.output_dir, "checkpoints"),
        auto_insert_metric_name=False,
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # gpu_count = torch.cuda.device_count()

    # Configure Trainer
    trainer = pl.Trainer.from_argparse_args(
        config.args, # type: ignore
        accelerator='gpu', devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        auto_lr_find=config.auto_lr,
        default_root_dir=config.output_dir,
        logger=configure_logger(config),
        profiler="advanced"
    )

    # 加载 checkpoint 测试
    if config.test_from_ckpt:
        ckpt_path = None
        if config.test_opt1 == "last":
            assert config.last_model_path is not None, "last_model_path is None"
            ckpt_path = config.last_model_path
        elif config.test_opt1 == "best" or config.test_opt1 is None:
            ckpt_path = config.best_model_path

        # cprint.info("Test from checkpoint: ", ckpt_path)
        if ckpt_path:
            trainer.test(ckpt_path=ckpt_path, model=theta, datamodule=data)

    else:

        if config.auto_lr:
            print(cp.yellow("Auto LR Finding..."))
            trainer.tune(theta, datamodule=data)  # with auto_lr_find=True

        trainer.fit(theta, datamodule=data)
        config.save_best_model_path(model_checkpoint.best_model_path)
        config.save_config()

        if model_checkpoint.best_model_path:
            trainer.test(theta, datamodule=data, ckpt_path=model_checkpoint.best_model_path)
        wandb.finish(quiet=True)

    result = {
        # "best_model_path": model_checkpoint.best_model_path,
        "best_f1": theta.best_f1,
        "test_f1": theta.test_f1,
        "test_f1*": theta.test_f1_plus,
        "test_p": theta.test_p,
        "test_r": theta.test_r,
        "ner_f1": theta.ner_f1,
        # "ner_p": theta.ner_p,
        # "ner_r": theta.ner_r,
        "rel_f1": theta.rel_f1,
        # "rel_p": theta.rel_p,
        # "rel_r": theta.rel_r,
        # "final_config": config.final_config
    }

    config.save_result(result)
    return result


def configure_logger(config):
    """ TensorBoardLogger (offline) or WandbLogger (Online) """

    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logger = pl.loggers.WandbLogger( # type: ignore
        project="Filter",
        name=config.tag,
        save_dir=config.output_dir,
        offline=config.offline,
        reinit=True,
        save_code=True,)

    logger.log_hyperparams(config)
    return logger


if __name__ == "__main__":
    print(main())

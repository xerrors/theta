import argparse

import pytorch_lightning as pl

import utils

from xerrors import cprint as cp


def setup_parser(func_mode, **kwargs):
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    # More: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser]) # type: ignore

    parser.add_argument("--task", type=str, default="ere", choices=["ner", "rc", "ere"])

    # Basic arguments
    parser.add_argument("--gpu", type=str, default="not specified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="output")

    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-config", type=str, default="models/bert.yaml")
    parser.add_argument("--dataset-config", type=str, default="datasets/ace2005/ace2005.yaml")

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--test-batch-size", type=int, default=0)
    parser.add_argument("--num-worker", type=int, default=8)
    parser.add_argument("--tag", type=str, default="debug")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-borther-confirm", action="store_true", default=False)

    parser.add_argument("--test-from-ckpt", type=str, default=None)


    # 函数模式，比较特殊，是没有办法使用命令行参数的，所以全部使用默认参数并由 kwargs 更新
    if func_mode:
        known_args = parser.parse_args(args=[])
        default_args = vars(known_args)
        for key, value in kwargs.items():
            if key in default_args and default_args[key] != value:
                default_args[key] = value

        return known_args

    return parser.parse_args()

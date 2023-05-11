from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import utils

no_decay_param = ["bias", "LayerNorm.weight", "layer_norm.weight"]


def get_optimizer(theta, config):
    """configure optimizer and lr_scheduler
    Args:
        theta (Theta): model
        config (Config): config
    """

    model_lr = config.get("model_lr", config.lr)  # default model_lr = lr
    task_lr = config.get("task_lr", config.lr)
    decoder_lr = config.get("decoder_lr", config.lr)

    optimizer_group_parameters = []
    optimizer_group_parameters.extend(get_params(theta, name="plm_model", lr=model_lr))
    optimizer_group_parameters.extend(get_params(theta, name="filter", lr=task_lr))
    optimizer_group_parameters.extend(get_params(theta, name="span_ner", lr=task_lr))
    optimizer_group_parameters.extend(get_params(theta, name="ner_model", lr=task_lr))
    optimizer_group_parameters.extend(get_params(theta, name=None, lr=decoder_lr, added_list=["plm_model", "filter", "span_ner", "ner_model"]))

    optimizer = AdamW(optimizer_group_parameters, lr=config.lr, eps=1e-8)

    if config.warmup:
        num_training_steps = calc_num_training_steps(theta)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps*config.warmup,
            num_training_steps=num_training_steps
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    else:
        return optimizer


def get_params(model, name, lr, exclude=[], added_list=[]):
    """get optimizer group ny model mame

    Args:
        name (str): model name or can be set to None, it means all parems that not be added.
        lr (float): learning_rate
        exclude (list, optional): exclude. Defaults to [].

    Returns:
        list: optimizer group parameter

    Filter:
        能被判定为需要添加的参数
        1. 没有指定 name 或者 name 在参数全名中 比如 name = "roberta"，则会添加所有包含 roberta 的参数
        2. 名字不存在于已经添加的参数列表中 比如已经添加了 "roberta"，则不会再添加包含 "roberta" 的参数
        3. 名字不存在于被排除的参数列表中 比如 exclude = ["bias"]，则不会添加包含 "bias" 的参数
        4. 关于 decay 的过滤, with_decay 为 True 时，参数名中不包含 no_decay_param 中的任意一个
    """

    global no_decay_param

    def filter(full_name, params, with_decay): return (not name or name in full_name) \
        and not any(pram_name in full_name for pram_name in exclude) \
        and not any(pram_name in full_name for pram_name in added_list) \
        and params.requires_grad \
        and (with_decay ^ any(pram_name in full_name for pram_name in no_decay_param))

    # 控制台输出信息（仅仅是为了方便查看）
    params_info = ""
    for n, p in model.named_parameters():
        if filter(n, p, True) and ".encoder.layer" not in n:
            params_info += f"\n- {n}, {list(p.size())}"
    if params_info != "":
        print(f"\nParameters (execpt encoder.layer) LR: {lr:.2e}{params_info}")

    # 排除 weight_decay
    with_decay = [p for n, p in model.named_parameters() if filter(n, p, True)]
    without_decay = [p for n, p in model.named_parameters() if filter(n, p, False)]
    optimizer_group = [
        {"params": with_decay, 'lr': lr, "weight_decay": 0.01},
        {"params": without_decay, 'lr': lr, "weight_decay": 0}
    ]

    if optimizer_group[0]["params"] == []:
        if name:
            print(utils.yellow("[WARNING]"),
                  f"{name} that can not be added to optimizer group.")
        return []

    return optimizer_group


def calc_num_training_steps(theta):
    """Total training steps inferred from datamodule and devices."""

    trainer = theta.trainer
    # calc dataset size
    if isinstance(trainer.limit_train_batches, int) and trainer.limit_train_batches != 0:
        dataset_size = trainer.limit_train_batches
    elif isinstance(trainer.limit_train_batches, float):
        # limit_train_batches is a percentage of batches
        dataset_size = len(trainer.datamodule.train_dataloader())
        dataset_size = int(dataset_size * trainer.limit_train_batches)
    else:
        dataset_size = len(trainer.datamodule.train_dataloader())

    # calc num devices
    num_devices = max(1, trainer.num_devices)

    # calc max steps
    effective_batch_size = trainer.accumulate_grad_batches * num_devices
    max_estimated_steps = (
        dataset_size // effective_batch_size) * trainer.max_epochs

    # check if max steps is limited by the user or system
    if trainer.max_steps > 0 and trainer.max_steps < max_estimated_steps:
        return trainer.max_steps

    return max_estimated_steps

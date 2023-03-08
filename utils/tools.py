import os

from transformers import logging

from .cprint import yellow, blue


def config_logging(config):
    """设置日志等级以及配置保存的路径"""
    if config.debug:
        logging.set_verbosity_debug()
    else:
        logging.set_verbosity_error()

    # logging.basicConfig(
    #         filename=os.path.join(config.output_dir, f"{config.start}-{config.tag}.log"),
    #         format='%(asctime)s > %(message)s',
    #         datefmt='%Y/%m/%d %I:%M:%S',
    #         level=logging.DEBUG,
    #         encoding='utf-8')


def get_gpu_by_user_input():

    try:
        os.system("gpustat")
    except:
        print(yellow("WARNING:"),
              "Try to install gpustat to check GPU status: pip install gpustat")

    gpu = input("\nSelect GPU >>> ")

    assert gpu and int(gpu) in [0, 1, 2, 3], \
        "Can not run scripts on GPU: {}".format(gpu if gpu else "None")
    print("This scripts will use GPU {}".format(gpu))
    return gpu


def confirm_value(key, value):
    """实现一个函数，函数接收一个默认值，让用户输入此次任务的tag，如果直接回车，则返回默认值"""
    if value:
        new_value = input(f"\nPlease confirm {key} (default: {blue(value)}) >>> ")
        if new_value:
            value = new_value

    assert value, "This value can not be empty!"

    return value


def confirm_bool(key, value):
    new_value = "none"

    while new_value not in ["y", "n", ""]:
        new_value = input(f"\nPlease confirm {key} (default: {blue(value)}) [y]/n >>> ")

    new_value = True if not new_value or new_value == "y" else False
    return new_value
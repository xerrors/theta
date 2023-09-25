from argparse import Namespace
import os
import time
import torch
import yaml
import json
import utils

from transformers import AutoConfig


class SimpleConfig(dict):

    def __key(self, key):
        return "" if key is None else key.lower()

    def __str__(self):
        return json.dumps(self)

    def __setattr__(self, key, value):
        self[self.__key(key)] = value

    def __getattr__(self, key):
        return self.get(self.__key(key))

    def __getitem__(self, key):
        return super().get(self.__key(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)


class Config(SimpleConfig):

    def __init__(self, args, **kwargs):
        """初始化配置

        配置的优先级：
            args < config < ext_config
        """

        ckpt = args.test_from_ckpt or kwargs.get("test_from_ckpt")
        if ckpt:
            self.load_from_ckpt(ckpt) # type: ignore
            # self.wandb = False
            self.offline = True

            self.ext_config = kwargs
            self.args = Namespace(**self.formatted_args) # type: ignore
            self.model = SimpleConfig(self.model)
            self.dataset = SimpleConfig(self.dataset)
            self.__load_ext_config()
            if kwargs.get("gpu") == "not specified" or kwargs.get("gpu") is None:
                self.gpu = utils.get_gpu_by_user_input()

            self.test_from_ckpt = ckpt
            self.last_test_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.test_result = os.path.join(self.output_dir, f"test-result-{self.tag}-{self.last_test_time}.json")
            self.prepare()
            return

        self.args = args
        self.ext_config = kwargs

        # 1. 从 args 中加载配置
        self.__load_config_from_args()
        self.__load_config(args.config)

        # 2. 加载 model 和 dataset 的配置
        self.model = self.parse_config(self.model_config, "model")
        self.dataset = self.parse_config(self.dataset_config, "dataset")

        # 3. 从 ext_config 中加载配置
        self.__load_ext_config()
        self.__replace_config_to_args()

        # 4. 将配置文件保存到实验输出文件夹里面
        self.handle_config()
        self.prepare()

        # 开始运行之前保存配置
        self.save_config("config.pre.yaml")

        # 创建一个快捷方式，类型是文件夹，指向 output_dir，如果此快捷方式已经存在就删除重新创建快捷方式
        link = os.path.join(self.output, "latest") # type: ignore
        # if os.path.exists(link) and os.path.islink(link):
        #     os.remove(link)

        if not self.debug:
            dir_name = self.output_dir.split(os.sep)[-1]
            try:
                os.symlink(f"{dir_name}", link, target_is_directory=True)
            except:
                os.remove(link)
                os.symlink(f"{dir_name}", link, target_is_directory=True)

    def parse_config(self, config_file, config_type):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        parsed_config = SimpleConfig(config)

        if config_type == "model":
            model_config = AutoConfig.from_pretrained(parsed_config.model_name_or_path) # type: ignore
            parsed_config.update(model_config.to_dict())

        return parsed_config

    def handle_config(self):
        # 确认 tag，如果是函数传参调用的模式，就不需要确认 tag
        if not self.with_ext_config and not self.no_borther_confirm:
            self.tag = utils.confirm_value("tag", self.tag)

            if self.debug and self.wandb and not self.offline:
                self.offline = utils.confirm_bool("offline with wandb", self.offline)

        # 创建此次实验的基本信息，运行时间，输出路径
        self.start = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.output_dir = os.path.join(str(self.output), f"ouput-{self.start}-{self.tag}")
        os.makedirs(self.output_dir, exist_ok=True)

        if self.debug and not self.test_from_ckpt:
            self.output_dir = os.path.join(str(self.output), "debug", f"{self.start}-{self.tag}")
            os.makedirs(self.output_dir, exist_ok=True)

        # 快速验证模式
        if self.fast_dev_run:
            # self.wandb = False
            self.offline = True
            print(utils.blue_background(">>> FAST DEV RUN MODE <<<"))

        self.test_result = os.path.join(self.output_dir, f"test-result.json")

    def prepare(self):
        # 设置日志等级
        utils.config_logging(self)

        # 配置 GPU
        if self.gpu == "not specified":
            self.gpu = utils.get_gpu_by_user_input()
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        if self.precision == 16:
            torch.set_float32_matmul_precision('medium')
        else:
            torch.set_float32_matmul_precision('high')

    def save_best_model_path(self, path):
        self.best_model_path = path
        self.last_model_path = path.replace(path.split(os.sep)[-1], "last.ckpt")
        print(utils.green("Done!"), f"Best model saved at: {path}")

    def save_config(self, filename="config.yaml"):
        """将自身的所有属性都保存下来，不包含方法"""
        config = {}
        self.formatted_args = SimpleConfig(vars(self.args))
        for key, value in self.items():
            if isinstance(value, SimpleConfig):
                items = {}
                for k, v in value.items():
                    if not k.startswith("__") and not callable(v):
                        items[k] = v
                config[key] = items

            elif not key.startswith("__") and not callable(value) and not key == "args":
                config[key] = value

        file_path = os.path.join(self.output_dir, filename)
        # print(utils.green("Done!"), f"Config saved at: {file_path}")
        with open(file_path, 'w') as f:
            yaml.dump(config, f)

        self.final_config = file_path

    def save_result(self, result:dict):
        with open(self.test_result, 'w') as f:
            json.dump(result, f, indent=4)

    def load_from_ckpt(self, ckpt_path):
        # 从 ckpt 中读取
        with open(ckpt_path, 'r') as f:
            ckpt = yaml.load(f, Loader=yaml.FullLoader)

        for key, value in ckpt.items():
            self.__setattr__(key, value)

    def __load_config_from_args(self):
        for key, value in vars(self.args).items():
            self.__setattr__(key, value)

    def __replace_config_to_args(self):

        # 约定 batch_size
        if self.global_batch_size:
            self.accumulate_grad_batches = self.global_batch_size // self.batch_size
        else:
            self.accumulate_grad_batches = 1

        for key, value in vars(self.args).items():
            if key in self.keys():
                setattr(self.args, key, self[key])

    def __load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        for key, value in config.items():
            self.__setattr__(key, value)

    def __load_ext_config(self):
        """ 优先级最高 """
        if len(self.ext_config) == 0:
            return

        self.with_ext_config = True
        for key, value in self.ext_config.items():
            self.__setattr__(key, value)

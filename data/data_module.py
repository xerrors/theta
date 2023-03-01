import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

import utils
from data.data_structures import Dataset
from data.utils import convert_dataset_to_samples


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name_or_path)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        # self.tokenizer.add_special_tokens({'additional_special_tokens': gen_tokens(config)})

        self.rel_num = len(config.dataset.rels)
        self.id2rel = config.dataset.rels
        self.rel2id = {name: idx for idx,
                       name in enumerate(config.dataset.rels)}

        self.ner_num = len(config.dataset.ners)
        self.id2ner = config.dataset.ners
        self.ner2id = {name: idx for idx,
                       name in enumerate(config.dataset.ners)}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.data_train = self.__get_dataset("train")
            self.data_val = self.__get_dataset("val")
        if stage == "test" or stage is None or stage == "predict":
            self.data_test = self.__get_dataset("test")

    def __get_dataset(self, mode):
        """根据不同的任务类型以及数据集类型使用不同的数据加载方法"""
        print(utils.green(f"Loading {mode} data..."))
        if self.config.dataset.name in ["ace2005"]:
            datasets = self.get_dataset_ace(mode)
        else:
            raise NotImplementedError(
                f"Dataset {self.config.dataset.name} not implemented!")

        return datasets

    def get_dataset_ace(self, mode):

        dataset = Dataset(self.config.dataset[mode])
        features = convert_dataset_to_samples(
            dataset, self.config, self.tokenizer, is_test=(mode == "test"))

        # 由于这里是已经将 rel 转化为 map 的形式，如果出现主机内存爆掉的情况，就自己写一个 dataloader，等到加载的时候再转化为 map
        dataset = TensorDataset(
            features["input_ids"],
            features["attention_mask"],
            features["ner_maps"],
            features["rel_maps"],
            features["ent_corres"],
            features["pos"],)

        return dataset

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.config.batch_size, num_workers=self.config.num_worker, pin_memory=True)

    def val_dataloader(self):
        batch_size = self.config.batch_size if not self.config.test_batch_size else self.config.test_batch_size
        return DataLoader(self.data_val, shuffle=False, batch_size=batch_size, num_workers=self.config.num_worker, pin_memory=True)

    def test_dataloader(self):
        batch_size = self.config.batch_size if not self.config.test_batch_size else self.config.test_batch_size
        return DataLoader(self.data_test, shuffle=False, batch_size=batch_size, num_workers=self.config.num_worker, pin_memory=True)

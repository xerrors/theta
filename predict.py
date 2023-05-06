from utils.args import setup_parser
from models.theta import Theta
from config import Config
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from data.data_module import DataModule

from xerrors import cprint as cp

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


class RelationExtractionModel:

    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        self.load_model() 
        self.load_data()
        
    def load_model(self):

        args = setup_parser(func_mode=True, test_from_ckpt=self.ckpt_path)
        config = Config(args, test_from_ckpt=self.ckpt_path)  # configure logging, save config, gpu etc.
        seed_every_thing(config.seed)

        data = DataModule(config)  # The data
        ckpt_model = config.best_model_path
        assert ckpt_model is not None, "No checkpoint model found."
        
        model = Theta.load_from_checkpoint(ckpt_model, config=config, data=data)
        model.cuda()
        model.eval()

        self.config = config
        self.data = data
        self.model = model

        cp.success("Predict", "Model loaded.")

    def load_data(self):
        self.dataset = self.data.get_dataset_ace_for_predict()

    def get_instance(self, idx=15):
        item = self.dataset[idx]
        return item

    def predict(self, sent):
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                cp.error("Predict", "Model not loaded.")
                return None

        output = self.model.predict_step(sent)
        return output



if __name__ == "__main__":

    model = RelationExtractionModel("output/ouput-2023-05-05_04-11-07-Omicron-AttnE-NER-mlp/config.yaml")
    cp.success("Predict", "Model loaded.")

    item = model.get_instance(16)
    cp.info("Predict", "Instance loaded.")

    output = model.predict(item["sent"])
    cp.info("Predict", "Prediction done.")
    cp.print_json(output)

    # Gold
    triples = []
    for rel in item["relations"]:
        triples.append((rel["subject"], rel["object"], rel["relation"]))
    print("Gold triples:", triples)

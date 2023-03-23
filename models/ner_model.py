import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchcrf import CRF

from models.components import MultiNonLinearClassifier


class NERModel(pl.LightningModule):

    def __init__(self, config, ent_ids, lmhead):
        super().__init__()
        self.config = config
        self.ent_ids = ent_ids

        if config.use_ner == "lmhead":
            self.lmhead = lmhead
        else:
            self.classifier = MultiNonLinearClassifier(config.model.hidden_size, len(ent_ids))

        if self.config.use_crf:
            self.crf = CRF(len(ent_ids), batch_first=True)

        self.num_ent_type = len(self.config.dataset.ents)

    def forward(self, hidden_output, pos=None, labels=None):

        if self.config.use_ner == "lmhead":
            logits = self.lmhead(hidden_output)
            logits = logits[..., self.ent_ids]
        else:
            logits = self.classifier(hidden_output)

        loss = None
        bsz = logits.shape[0]
        if labels is not None:

            if self.config.use_crf:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for b in range(labels.shape[0]):
                    mask[b, pos[b,0]:pos[b,1]] = 1
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')

            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                # Only keep active parts of the loss
                if pos is not None:
                    labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                    labels = labels.view(-1)
                    new_logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                    new_logits = new_logits.view(-1, len(self.ent_ids))
                else:
                    new_logits = logits.view(-1, len(self.ent_ids))
                    labels = labels.view(-1)

                loss = loss_fct(new_logits, labels)

        return logits, loss

    def decode_entities(self, logits, pos=None):
        """return 左闭右开 [[(start, end, type), (...)],[],[(...)]]"""
        if logits.shape[-1] == len(self.ent_ids):
            if self.config.use_crf:
                logits = torch.tensor(self.crf.decode(logits), device=logits.device)
            else:
                logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz = logits.shape[0]
        # O B*7(1~8) I*7(8-15), self.num_ent_type = 7
        for b in range(bsz):
            entity = []
            start = False
            sent_start, sent_end = pos[b,0], pos[b,1]
            for i in range(sent_start, sent_end):
                # 判断是否是 B 标签
                if logits[b, i] > 0 and logits[b, i] <= self.num_ent_type:
                    start = True
                    entity.append([i, i+1, logits[b, i].item()-1])
                # 判断是否是 I 标签
                elif logits[b, i] > self.num_ent_type and start:
                    entity[-1][1] = i + 1 # 左闭右开
                else:
                    start = False

            entities.append(entity)

        return entities

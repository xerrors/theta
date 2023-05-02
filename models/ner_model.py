import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchcrf import CRF

from models.components import MultiNonLinearClassifier, SelfAttention
from models.functions import getPretrainedLMHead


class NERModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()
        self.config = theta.config
        self.ent_ids = theta.ent_ids

        config = self.config
        hidden_size = config.model.hidden_size

        if config.use_ner == "lmhead":
            self.lmhead = getPretrainedLMHead(theta.plm_model, config.model)

        elif config.use_ner == "linear":
            self.classifier = nn.Linear(hidden_size, len(self.ent_ids))

        else:
            self.classifier = MultiNonLinearClassifier(hidden_size, len(self.ent_ids), layers_num=2)

        if self.config.use_crf:
            self.crf = CRF(len(self.ent_ids), batch_first=True)

        self.self_attn = SelfAttention(embed_dim=hidden_size)

        self.num_ent_type = len(self.config.dataset.ents)

    def forward(self, hidden_state, labels=None, graph=None, mask=None):

        if self.config.use_ent_attn:
            hidden_state = self.self_attn(hidden_state)

        if graph is not None:
            hidden_state = graph.query_ents(hidden_state)

        if mask is None:
            mask = torch.ones(hidden_state.shape[:2], device=hidden_state.device)

        if self.config.use_ner == "lmhead":
            assert self.lmhead is not None
            logits = self.lmhead(hidden_state)
            logits = logits[..., self.ent_ids]
        else:
            logits = self.classifier(hidden_state)

        loss = torch.tensor(0.0, device=logits.device)
        bsz = logits.shape[0]

        if labels is not None:

            if self.config.use_crf:
                loss = -self.crf(logits, labels, mask=mask, reduction='token_mean')

            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                new_logits = logits.view(-1, len(self.ent_ids))
                new_labels = labels.view(-1).long()

                loss = (loss_fct(new_logits, new_labels) * mask.view(-1)).sum() / mask.sum()

        return logits, loss

    def decode_entities(self, logits, pos=None):
        """return 左闭右开 [[(start, end, type), (...)],[],[(...)]]"""
        if len(logits.shape) == 3 and logits.shape[-1] == len(self.ent_ids):
            if self.config.use_crf:
                logits = torch.tensor(self.crf.decode(logits), device=logits.device)
            else:
                logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz, seq_len = logits.shape[0], logits.shape[1]
        # O B*7(1~8) I*7(8-15), self.num_ent_type = 7
        for b in range(bsz):
            entity = []
            start = False

            if pos is not None:
                sent_start, sent_end = pos[b,0], pos[b,1]
            else:
                sent_start, sent_end = 0, seq_len

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


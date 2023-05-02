import itertools
import math
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.components import MultiNonLinearClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


class FilterModel(pl.LightningModule):

    def __init__(self, theta) -> None:
        super().__init__()
        self.config = theta.config
        self.log = theta.log
        self.pre_mode = None
        self.pred = None
        self.labels = None

        self.filter_entity_pair_net = MultiNonLinearClassifier(self.config.model.hidden_size * 2, 1, layers_num=2)

        self.sub_proj = nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)
        self.obj_proj = nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)

        self.hard_filter_table = torch.load("datasets/ace2005/ent_rel_corres.data").sum(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def convert_bij_to_index(self, bij, entities):
        """找到 batch i 中的第 i 个实体和第 j 个实体在 logits 里面的位置
        因为 logits 是将不同batch 的不同大小的实体对应表展开之后的，所以需要这样一个映射函数
        """
        batch_num = [len(e) for e in entities]
        batch_sqrt = [len(e) * len(e) for e in entities]
        batch_start = [0] + list(itertools.accumulate(batch_sqrt))[:-1]

        b, i, j = bij
        index = batch_start[b] + i * batch_num[b] + j

        return index

    def get_filter_label(self, entities, triples, logits, map_dict):
        labels = torch.zeros_like(logits)
        for b, triple in enumerate(triples):
            for t in triple:
                if t[0] == -1:
                    continue
                i = map_dict.get((b, t[0].item()))
                j = map_dict.get((b, t[2].item()))
                if i is None or j is None:
                    continue
                index = self.convert_bij_to_index((b,i,j), entities)
                labels[index] = 1
                if self.config.use_filter_label_enhance:
                    index = self.convert_bij_to_index((b,j,i), entities)
                    labels[index] = 1
        return labels

    def forward(self, hidden_state, entities, triples=None, mode="train"):

        logits = []
        map_dict = {}

        hidden_state = self.dropout(hidden_state)

        for i in range(len(entities)):
            if len(entities[i]) == 0: continue

            # 记录实体 e 在此 batch i 的所有实体中的位置，后面需要用到表格索引的
            for j, e in enumerate(entities[i]):
                map_dict[(i, e[0])] = j

            if not self.config.use_ent_hidden_state or self.config.use_ent_hidden_state == "head":
                ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])
            elif self.config.use_ent_hidden_state == "add":
                head_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])
                tail_hs = torch.stack([hidden_state[i, ent[1]-1] for ent in entities[i]])
                ent_hs = head_hs + tail_hs
            elif self.config.use_ent_hidden_state == "mean":
                ent_hs = torch.stack([hidden_state[i, ent[0]:ent[1]].mean(dim=0) for ent in entities[i]])
            elif self.config.use_ent_hidden_state == "max":
                ent_hs = torch.stack([hidden_state[i, ent[0]:ent[1]].max(dim=0)[0] for ent in entities[i]])
            else:
                raise NotImplementedError("use_ent_hidden_state: {}".format(self.config.use_ent_hidden_state))

            # ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])    # (ent_num, hidden_size)
            ent_num, hidden_size = ent_hs.shape

            ent_hs_x = self.sub_proj(ent_hs)
            ent_hs_y = self.obj_proj(ent_hs)

            if not self.config.use_filter_opt1 or self.config.use_filter_opt1 == "attention":
                ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(-2, -1)) / math.sqrt(hidden_size)    # (ent_num, ent_num, hidden_size)

            elif self.config.use_filter_opt1 == "attention_softmax":
                ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(-2, -1)) / math.sqrt(hidden_size)    # (ent_num, ent_num, hidden_size)
                ent_hs_pair = ent_hs_pair.softmax(dim=-1)

            elif self.config.use_filter_opt1 == "concat":
                ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
                ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1)    # (ent_num, ent_num, hidden_size * 2)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)

            elif self.config.use_filter_opt1 == "concat_softmax":
                ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
                ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)
                ent_hs_pair = ent_hs_pair.softmax(dim=-1)

            else:
                raise ValueError("use_filter_opt1 must be in [None, 'attention', 'attention_softmax', 'concat', 'concat_softmax']")

            ent_hs_pair = ent_hs_pair.view(-1, 1).squeeze(-1)    # (ent_num, ent_num, hidden_size * 2)

            logits.append(ent_hs_pair)

        logits = torch.cat(logits, dim=0)    # (batch_size * ent_num * ent_num,)
        logits = logits.sigmoid()

        loss = torch.tensor(0.0, device=logits.device)
        if triples is not None:
            # sub_s, sub_e, obj_s, obj_e, rel_id, sub_type, obj_type
            labels = self.get_filter_label(entities, triples, logits, map_dict)

            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)

            if mode == 'train':
                # 将 logits > 0.5 的位置置为 1，否则置为 0，然后与 labels 计算 f1
                pred = torch.where(logits > 0.5, torch.ones_like(logits), torch.zeros_like(logits))
                self.pred = pred if self.pred is None else torch.cat([self.pred, pred], dim=0)
                self.labels = labels if self.labels is None else torch.cat([self.labels, labels], dim=0)

            elif self.pre_mode == 'train' and self.pred is not None and self.labels is not None:
                labels = self.labels.cpu().numpy()
                pred = self.pred.cpu().numpy()
                self.log("info/filter_f1", float(f1_score(labels, pred, zero_division=0)))
                self.log("info/filter_precision", float(precision_score(labels, pred, zero_division=0)))
                self.log("info/filter_recall", float(recall_score(labels, pred, zero_division=0)))
                self.pred = None
                self.labels = None
            else:
                pass

        self.pre_mode = mode
        return logits, loss, map_dict

    def get_draft_ent_groups(self, entities, batch_idx, map_dict, logits, mode):
        '''Get the draft entity groups for each entity in the batch.

        Args:
            entities (list): list of entities in the batch
            batch_idx (int): batch index
            map_dict (dict): map dict
            sort (str, optional): sort method as "score", "gold". Defaults to "score".
        '''

        ent_groups = []

        pairs = list(itertools.permutations(entities[batch_idx], 2))
        if len(pairs) > 0:
            random.shuffle(pairs)

            for sub_pos, obj_pos in pairs:
                i = map_dict[(batch_idx, sub_pos[0])]
                j = map_dict[(batch_idx, obj_pos[0])]
                if i == j:
                    continue
                index = self.convert_bij_to_index((batch_idx, i, j), entities)
                score = logits[index].item()

                if self.config.use_filter_hard and mode != 'train':
                    sub_type, obj_type = sub_pos[2], obj_pos[2]
                    if self.hard_filter_table[sub_type, obj_type] == 0:
                        continue

                ent_groups.append((sub_pos, obj_pos, score))
            ent_groups = sorted(ent_groups, key=lambda a : a[-1], reverse=True)

        return ent_groups

    def hard_filter(self):
        pass
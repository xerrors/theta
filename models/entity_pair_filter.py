import itertools
import math
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

        self.filter_entity_pair_net = MultiNonLinearClassifier(self.config.model.hidden_size * 2, 1)

        self.sub_proj = nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)
        self.obj_proj = nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)

        self.hard_filter_table = torch.load("datasets/ace2005/ent_rel_corres.data").sum(dim=-1)


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
        return labels

    def forward(self, hidden_state, entities, triples=None, mode="train"):

        logits = []
        map_dict = {}

        for i in range(len(entities)):
            if len(entities[i]) == 0: continue

            # 记录实体 e 在此 batch i 的所有实体中的位置，后面需要用到表格索引的
            for j, e in enumerate(entities[i]):
                map_dict[(i, e[0])] = j

            ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])    # (ent_num, hidden_size)
            ent_num, hidden_size = ent_hs.shape

            ent_hs_x = self.sub_proj(ent_hs)
            ent_hs_y = self.obj_proj(ent_hs)
            # torch.matmul(A, B) = A @ B
            ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(-2, -1)) / math.sqrt(hidden_size)    # (ent_num, ent_num, hidden_size)
            ent_hs_pair = ent_hs_pair.view(-1, 1).squeeze(-1)    # (ent_num, ent_num, hidden_size * 2)

            logits.append(ent_hs_pair)

        logits = torch.cat(logits, dim=0)    # (batch_size * ent_num * ent_num,)

        loss = torch.tensor(0.0, device=logits.device)
        if triples is not None:
            # sub_s, sub_e, obj_s, obj_e, rel_id, sub_type, obj_type
            labels = self.get_filter_label(entities, triples, logits, map_dict)

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

            if mode == 'train':
                # 将 logits > 0.5 的位置置为 1，否则置为 0，然后与 labels 计算 f1
                pred = torch.where(logits > 0.5, torch.ones_like(logits), torch.zeros_like(logits))
                f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy()) if pred.sum() != 0 else 0.0
                precision = precision_score(labels.cpu().numpy(), pred.cpu().numpy()) if pred.sum() != 0 else 0.0
                recall = recall_score(labels.cpu().numpy(), pred.cpu().numpy()) if labels.sum() != 0 else 0.0
                self.log("info/filter_f1", float(f1))
                self.log("info/filter_precision", float(precision))
                self.log("info/filter_recall", float(recall))

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

        for sub_pos, obj_pos in itertools.permutations(entities[batch_idx], 2):
            i = map_dict[(batch_idx, sub_pos[0])]
            j = map_dict[(batch_idx, obj_pos[0])]
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
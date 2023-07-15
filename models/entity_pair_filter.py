import itertools
import math
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.metrics import f1_score_simple

from xerrors import debug
from models.components import MultiNonLinearClassifier, SelfAttention
from sklearn.metrics import f1_score, precision_score, recall_score


class FilterModel(pl.LightningModule):

    def __init__(self, theta) -> None:
        super().__init__()
        self.config = theta.config
        self.enable = self.config.use_filter and self.config.filter_rate > 0
        self.log = theta.log
        # self.pre_mode = None
        self.pred = None
        self.labels = None
        self.pred_val = None
        self.labels_val = None

        dropout_rate = 0.3 if self.config.use_filter_pro else 0.1

        attention_out_dim = int(self.config.model.hidden_size * float(self.config.get("use_filter_opt4", 1.0)))
        self.tag_size = len(self.config.dataset.rels) + 1 if self.config.use_filter_opt1 == "concat_pro" else 1

        if self.config.use_filter_attn:
            self.attn = SelfAttention(self.config.model.hidden_size)

        remove_bias = self.config.use_filter_opt2 == "rm_bias"
        self.sub_proj = nn.Linear(self.config.model.hidden_size, attention_out_dim, bias=(not remove_bias))
        self.obj_proj = nn.Linear(self.config.model.hidden_size, attention_out_dim, bias=(not remove_bias))

        self.filter_entity_pair_net = MultiNonLinearClassifier(
            attention_out_dim * 2,
            dropout_rate=dropout_rate,
            hidden_dim=256 if self.config.use_filter_pro else None,
            tag_size=self.tag_size)

        self.hard_filter_table = torch.load("datasets/ace2005/ent_rel_corres.data").sum(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        # metrics
        self.train_metrics = {
            "f1": 0,
            "precision": 0,
            "recall": 0,
        }

        self.val_metrics = {
            "f1": 0,
            "precision": 0,
            "recall": 0,
        }

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
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        for b, triple in enumerate(triples):
            for t in triple:
                if t[0] == -1:
                    continue
                i = map_dict.get((b, t[0].item()))
                j = map_dict.get((b, t[2].item()))
                if i is None or j is None:
                    continue
                index = self.convert_bij_to_index((b,i,j), entities)
                labels[index] = t[4] + 1 if self.config.use_filter_opt1 == "concat_pro" else 1
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

            if not self.config.use_rel_opt2 or self.config.use_rel_opt2 == "head":
                ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])
            elif self.config.use_rel_opt2 == "add":
                head_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])
                tail_hs = torch.stack([hidden_state[i, ent[1]-1] for ent in entities[i]])
                ent_hs = head_hs + tail_hs
            elif self.config.use_rel_opt2 == "mean":
                ent_hs = torch.stack([hidden_state[i, ent[0]:ent[1]].mean(dim=0) for ent in entities[i]])
            elif self.config.use_rel_opt2 == "max":
                ent_hs = torch.stack([hidden_state[i, ent[0]:ent[1]].max(dim=0)[0] for ent in entities[i]])
            else:
                raise NotImplementedError("use_rel_opt2: {}".format(self.config.use_rel_opt2))

            if self.config.use_filter_attn:
                ent_hs = self.attn(ent_hs)

            # ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])    # (ent_num, hidden_size)
            ent_num, hidden_size = ent_hs.shape

            ent_hs_x = self.sub_proj(ent_hs)
            ent_hs_y = self.obj_proj(ent_hs)

            if self.config.use_filter_pro:
                ent_hs_x = self.dropout(ent_hs_x)
                ent_hs_y = self.dropout(ent_hs_y)

            if not self.config.use_filter_opt1 or self.config.use_filter_opt1 == "attention":
                ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(-2, -1)) / math.sqrt(hidden_size)    # (ent_num, ent_num, hidden_size)

            elif self.config.use_filter_opt1 == "concat" or self.config.use_filter_opt1 == "concat_pro":
                ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
                ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1)    # (ent_num, ent_num, hidden_size * 2)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)

            else:
                raise ValueError("use_filter_opt1 must be in ['attention', 'concat', 'concat_pro']")

            ent_hs_pair = ent_hs_pair.view(-1, self.tag_size).squeeze(-1)    # (ent_num, ent_num, hidden_size * 2)

            logits.append(ent_hs_pair)

        logits = torch.cat(logits, dim=0)    # (batch_size * ent_num * ent_num,)


        loss = torch.tensor(0.0, device=logits.device)
        if triples is not None:
            # sub_s, sub_e, obj_s, obj_e, rel_id, sub_type, obj_type
            labels = self.get_filter_label(entities, triples, logits, map_dict)

            if self.config.use_filter_opt1 == "concat_pro":
                reduction = 'sum' if self.config.use_filter_sum_loss else "mean"
                loss_fct = nn.CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(logits, labels.long())
                pred = torch.argmax(logits, dim=-1)

            else: # Default
                reduction = 'sum' if self.config.use_filter_sum_loss else "mean"
                loss_fct = nn.BCEWithLogitsLoss(reduction=reduction)
                loss = loss_fct(logits, labels.float())
                logits_sigmoid = logits.sigmoid()
                pred = torch.where(logits_sigmoid > 0.5, torch.ones_like(logits_sigmoid), torch.zeros_like(logits_sigmoid))

            if mode == 'train':
                self.pred = pred if self.pred is None else torch.cat([self.pred, pred], dim=0)
                self.labels = labels if self.labels is None else torch.cat([self.labels, labels], dim=0)
            elif mode == "dev" or mode == "test":
                self.pred_val = pred if self.pred_val is None else torch.cat([self.pred_val, pred], dim=0)
                self.labels_val = labels if self.labels_val is None else torch.cat([self.labels_val, labels], dim=0)

        # self.pre_mode = mode
        if self.config.use_filter_opt1 == "concat_pro":
            logits = logits.softmax(dim=-1)[:, 1:].sum(dim=-1)
        else:
            logits = logits.sigmoid()

        return logits, loss, map_dict

    def log_filter_train_metrics(self):
        if self.pred is not None and self.labels is not None:
            labels = self.labels.cpu().numpy()
            pred = self.pred.cpu().numpy()
            if self.config.use_filter_opt1 == "concat_pro":
                f1, p, r = f1_score_simple(labels, pred)
            else:
                f1 = f1_score(labels, pred, zero_division=0)
                p = precision_score(labels, pred, zero_division=0)
                r = recall_score(labels, pred, zero_division=0)

            self.train_metrics = {
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
            self.log("info/filter_f1", self.train_metrics["f1"])
            self.log("info/filter_precision", self.train_metrics["precision"])
            self.log("info/filter_recall", self.train_metrics["recall"])
            self.pred = None
            self.labels = None
            debug.log(self.config.debug, self.train_metrics["f1"])


    def log_filter_val_metrics(self):
        if self.pred_val is not None and self.labels_val is not None:
            labels = self.labels_val.cpu().numpy()
            pred = self.pred_val.cpu().numpy()
            if self.config.use_filter_opt1 == "concat_pro":
                f1, p, r = f1_score_simple(labels, pred)
            else:
                f1 = f1_score(labels, pred, zero_division=0)
                p = precision_score(labels, pred, zero_division=0)
                r = recall_score(labels, pred, zero_division=0)

            self.val_metrics = {
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
            self.log("info/filter_f1_val", self.val_metrics["f1"])
            self.log("info/filter_precision_val", self.val_metrics["precision"])
            self.log("info/filter_recall_val", self.val_metrics["recall"])
            self.pred_val = None
            self.labels_val = None
            debug.log(self.config.debug, self.val_metrics["f1"])


    def get_draft_ent_groups(self, entities, batch_idx, map_dict, logits, mode, pred=None):
        '''Get the draft entity groups for each entity in the batch.'''

        ent_groups = []

        pairs = list(itertools.permutations(entities[batch_idx], 2))
        if len(pairs) > 0:

            # some tricks
            if self.enable:
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

                if pred is not None:
                    pred_score = pred[index].item()
                    ent_groups.append((sub_pos, obj_pos, score, pred_score))

                else:
                    ent_groups.append((sub_pos, obj_pos, score))

            if pred is not None:
                ent_groups = sorted(ent_groups, key=lambda a : (a[2], a[3]), reverse=True)
                ent_groups = [g[:3] for g in ent_groups]

            else:
                ent_groups = sorted(ent_groups, key=lambda a : a[2], reverse=True)

        return ent_groups

    def hard_filter(self):
        pass
import itertools
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.metrics import f1_score_simple

from xerrors import debug
from models.components import MultiNonLinearClassifier, SelfAttention
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.Focal_Loss import focal_loss


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
        hidden_size = self.config.model.hidden_size

        dropout_rate = self.config.filter_dropout_rate # 0.1 default

        # attention_out_dim = int(self.config.model.hidden_size * float(self.config.get("use_filter_opt4", 1.0)))
        self.tag_size = len(self.config.dataset.rels) + 1
        self.get_bio_tag_embedding = lambda d:theta.plm_model.get_input_embeddings()(torch.tensor(theta.ent_ids, device=d))

        if self.config.use_length_embedding:
            self.length_embedding = theta.length_embedding
        # if self.config.use_filter_attn:

        #     self.attn_mode = self.config.use_filter_attn[0]
        #     self.attn_opt = self.config.use_filter_attn[1:]

        #     self.self_attn = SelfAttention(embed_dim=hidden_size)
        #     self.downscale = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        #     self.upscale = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        #     self.word_embeddings = theta.plm_model.get_input_embeddings()
        #     self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, bias=True, batch_first=True)
        #     self.layer_norm = nn.LayerNorm(hidden_size)

        # else:
        #     self.attn_mode = None
        #     self.attn_opt = None

        self.rel_ids = theta.rel_ids
        self.rel_type_num = len(self.config.dataset.rels)
        self.ent_type_num = len(self.config.dataset.ents)
        self.sub_proj = nn.Linear(hidden_size, hidden_size)
        self.obj_proj = nn.Linear(hidden_size, hidden_size)

        # if self.attn_opt == '2':
        #     classifier_input_dim = attention_out_dim
        # else:
        #     classifier_input_dim = attention_out_dim * 2

        self.filter_entity_pair_net = MultiNonLinearClassifier(
            hidden_size * 2,
            layers_num=self.config.filter_mlp_layer_num or 1,
            dropout_rate=dropout_rate,
            hidden_dim=None,
            tag_size=self.tag_size)

        if self.config.use_filter_sent_hs:
            self.sent_proj = nn.Linear(hidden_size, hidden_size)
            self.filter_entity_pair_net = MultiNonLinearClassifier(
                hidden_size * 3,
                layers_num=self.config.filter_mlp_layer_num or 1,
                dropout_rate=dropout_rate,
                hidden_dim=None,
                tag_size=self.tag_size)

            # 为了保证特征经过 linear 之后变化不大，参数初始化为
        if self.config.use_filter_proj_norm:
            nn.init.normal_(self.sub_proj.weight, std=0.02)
            nn.init.normal_(self.obj_proj.weight, std=0.02)

            if self.config.use_filter_sent_hs:
                nn.init.normal_(self.sent_proj.weight, std=0.02)

        self.hard_filter_table = torch.load("datasets/ace2005/ent_rel_corres.data").sum(dim=-1)
        self.dropout_hidden_state = nn.Dropout(0.1)

        self.loss_weight = torch.FloatTensor([self.config.get("na_filter_weight", 1)] + [1] * self.rel_type_num)

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
                labels[index] = t[4] + 1
                if self.config.use_filter_label_enhance:
                    index = self.convert_bij_to_index((b,j,i), entities)
                    labels[index] = 1
        return labels

    def forward(self, hidden_state, entities, triples=None, mode="train", current_epoch=None):

        logits = []
        map_dict = {}

        hidden_state = self.dropout_hidden_state(hidden_state)

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


            if self.config.use_length_embedding:
                length_embeds = self.length_embedding(torch.tensor([ent[1] - ent[0] for ent in entities[i]], device=hidden_state.device))
                ent_hs += length_embeds

            # if self.attn_opt == '1':
            #     tag_embeddings = self.word_embeddings(torch.tensor(self.rel_ids, device=hidden_state.device))
            #     tag_embeddings = tag_embeddings.unsqueeze(0) # [1, 1, ent_num, hidden_size]

            #     ent_hs = self.self_attn(ent_hs)
            #     attn_out = self.cross_attn(ent_hs.unsqueeze(0), tag_embeddings, tag_embeddings)[0]
            #     ent_hs = self.layer_norm(ent_hs + attn_out).squeeze(0)
            # if self.config.use_filter_tag == "b":
            #     ent_tag = torch.stack([self.get_bio_tag_embedding(hidden_state.device)[ent[2]+1] for ent in entities[i]])
            #     ent_hs += ent_tag

            # ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])    # (ent_num, hidden_size)
            ent_num, hidden_size = ent_hs.shape

            ent_hs_x = self.sub_proj(ent_hs)
            ent_hs_y = self.obj_proj(ent_hs)

            # if self.config.use_filter_tag == "a":
            #     ent_tag = torch.stack([self.get_bio_tag_embedding(hidden_state.device)[ent[2]+1] for ent in entities[i]])
            #     ent_hs_x += ent_tag
            #     ent_hs_y += ent_tag

            ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
            ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)

            if self.config.use_filter_sent_hs:
                sent_hs = hidden_state[i][0]
                sent_hs = sent_hs.unsqueeze(0).repeat(ent_num, ent_num, 1)
                ent_hs_pair = torch.cat([sent_hs, ent_hs_x, ent_hs_y], dim=-1)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)
            else:
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1)    # (ent_num, ent_num, hidden_size * 2)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)

            # if self.attn_opt == '3':
            #     tag_embeddings = self.word_embeddings(torch.tensor(self.rel_ids, device=hidden_state.device))
            #     tag_embeddings = tag_embeddings.unsqueeze(0) # [1, 1, ent_num, hidden_size]

            #     ent_hs_x = self.self_attn(ent_hs_x)
            #     attn_out = self.cross_attn(ent_hs_x.unsqueeze(0), tag_embeddings, tag_embeddings)[0]
            #     ent_hs_x = self.layer_norm(ent_hs_x + attn_out).squeeze(0)

            #     ent_hs_y = self.self_attn(ent_hs_y)
            #     attn_out = self.cross_attn(ent_hs_y.unsqueeze(0), tag_embeddings, tag_embeddings)[0]
            #     ent_hs_y = self.layer_norm(ent_hs_y + attn_out).squeeze(0)

            # if self.config.use_filter_pro:
            #     ent_hs_x = self.dropout(ent_hs_x)
            #     ent_hs_y = self.dropout(ent_hs_y)

            # if not self.config.use_filter_opt1 or self.config.use_filter_opt1 == "attention":
            #     ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(-2, -1)) / math.sqrt(hidden_size)    # (ent_num, ent_num, hidden_size)

            # elif self.config.use_filter_opt1 == "concat" or self.config.use_filter_opt1 == "concat_pro":
            #     ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
            #     ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)
            #     ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1)    # (ent_num, ent_num, hidden_size * 2)

                # if self.attn_opt == '2':
                #     tag_embeddings = self.word_embeddings(torch.tensor(self.rel_ids, device=hidden_state.device))
                #     tag_embeddings = tag_embeddings.unsqueeze(0) # [1, 1, ent_num, hidden_size]

                #     ent_hs_pair = self.downscale(ent_hs_pair)
                #     ent_hs_pair = self.self_attn(ent_hs_pair)
                #     ent_hs_pair = ent_hs_pair.view(1, -1, hidden_size)
                #     attn_out = self.cross_attn(ent_hs_pair, tag_embeddings, tag_embeddings)[0]
                #     ent_hs_pair = self.layer_norm(ent_hs_pair + attn_out)
                #     ent_hs_pair = ent_hs_pair.view(ent_num, ent_num, hidden_size)

            #     ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)

            # else:
            #     raise ValueError("use_filter_opt1 must be in ['attention', 'concat', 'concat_pro']")

            ent_hs_pair = ent_hs_pair.view(-1, self.tag_size).squeeze(-1)    # (ent_num, ent_num, hidden_size * 2)

            logits.append(ent_hs_pair)

        logits = torch.cat(logits, dim=0)    # (batch_size * ent_num * ent_num,)


        loss = torch.tensor(0.0, device=logits.device)
        if triples is not None:
            # sub_s, sub_e, obj_s, obj_e, rel_id, sub_type, obj_type
            labels = self.get_filter_label(entities, triples, logits, map_dict)

            # if self.config.use_filter_na_warm_up and mode == "train":
            #     assert current_epoch is not None, "current_epoch must be not None when use_filter_na_warm_up is True"
            #     sin_warm = lambda x: np.sin((min(1, x + 0.001) * np.pi / 2))
            #     self.loss_weight[0] = float(self.config.get("na_filter_weight", 1)) * sin_warm(current_epoch / int(self.config.use_filter_na_warm_up))

            if self.config.use_filter_focal_loss:
                scale_rate = int(self.config.use_filter_loss_sum) * 100
                assert scale_rate > 0, "use_filter_loss_sum must be greater than 0"
                loss_fct = focal_loss(alpha=self.config.use_focal_alpha, gamma=2, num_classes=self.tag_size, size_average=False)
                loss = loss_fct(logits, labels.long()) / scale_rate / self.config.batch_size * 16 / self.loss_weight.sum() * self.tag_size

            elif self.config.use_filter_loss_sum:
                scale_rate = int(self.config.use_filter_loss_sum)
                assert scale_rate > 0, "use_filter_loss_sum must be greater than 0"
                loss_fct = nn.CrossEntropyLoss(reduction='sum', weight=self.loss_weight.to(logits.device)) # , label_smoothing=0.1
                loss = loss_fct(logits, labels.long()) / scale_rate / self.config.batch_size * 16 #  / self.loss_weight.sum() * self.tag_size

            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean', weight=self.loss_weight.to(logits.device))
                loss = loss_fct(logits, labels.long())
            pred = torch.argmax(logits, dim=-1)

            # if self.config.use_filter_opt1 == "concat_pro":
            #     reduction = 'sum' if self.config.use_filter_sum_loss else "mean"
            #     loss_fct = nn.CrossEntropyLoss(reduction=reduction, weight=self.loss_weight.to(logits.device), label_smoothing=0.1)
            #     loss = loss_fct(logits, labels.long())
            #     pred = torch.argmax(logits, dim=-1)

            # else:
            #     reduction = 'sum' if self.config.use_filter_sum_loss else "mean"
            #     loss_fct = nn.BCEWithLogitsLoss(reduction=reduction, weight=self.loss_weight.to(logits.device))
            #     loss = loss_fct(logits, labels.float())
            #     logits_sigmoid = logits.sigmoid()
                # pred = torch.where(logits_sigmoid > 0.5, torch.ones_like(logits_sigmoid), torch.zeros_like(logits_sigmoid))

            if mode == 'train':
                self.pred = pred if self.pred is None else torch.cat([self.pred, pred], dim=0)
                self.labels = labels if self.labels is None else torch.cat([self.labels, labels], dim=0)
            elif mode == "dev" or mode == "test":
                self.pred_val = pred if self.pred_val is None else torch.cat([self.pred_val, pred], dim=0)
                self.labels_val = labels if self.labels_val is None else torch.cat([self.labels_val, labels], dim=0)

        # self.pre_mode = mode
        logits = logits.softmax(dim=-1)[:, 1:].sum(dim=-1)
        # if self.config.use_filter_opt1 == "concat_pro":
        #     logits = logits.softmax(dim=-1)[:, 1:].sum(dim=-1)
        # else:
        #     logits = logits.sigmoid()

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
            self.log("filter/f1", self.train_metrics["f1"])
            self.log("filter/precision", self.train_metrics["precision"])
            self.log("filter/recall", self.train_metrics["recall"])
            self.pred = None
            self.labels = None
            debug.log(self.config.debug, self.train_metrics["f1"])


    def log_filter_val_metrics(self):
        """Fake validation metrics, just for logging."""
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
            self.log("filter/f1_val", self.val_metrics["f1"])
            self.log("filter/precision_val", self.val_metrics["precision"])
            self.log("filter/recall_val", self.val_metrics["recall"])
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
                if self.config.use_negative == "refine":
                    ent_groups = sorted(ent_groups, key=lambda a : (bool(a[2]), a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups]
                elif self.config.use_negative == "simple":
                    ent_groups = sorted(ent_groups, key=lambda a : (bool(a[2]), -a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups]
                elif self.config.use_negative == "random":
                    ent_groups = sorted(ent_groups, key=lambda a : bool(a[2]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups]
                elif self.config.use_negative == "random2":
                    ent_groups = sorted(ent_groups, key=lambda a : bool(a[2]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups if g[3] > 1e-6]
                elif self.config.use_negative == "noise":
                    metric = self.train_metrics.get("f1", 0)
                    noise = np.random.normal(0, 1 - metric, len(ent_groups))
                    ent_groups_with_noise = [(g[0], g[1], g[2], g[3] + noise[i]) for i, g in enumerate(ent_groups)]
                    ent_groups_with_noise = sorted(ent_groups_with_noise, key=lambda a : (bool(a[2]), a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups_with_noise]
                elif self.config.use_negative == "noise2":
                    metric = self.train_metrics.get("f1", 0)
                    noise = np.random.normal(0, 1 - metric, len(ent_groups))
                    ent_groups_with_noise = [(g[0], g[1], g[2], g[3] + noise[i]) for i, g in enumerate(ent_groups)]
                    ent_groups_with_noise = sorted(ent_groups_with_noise, key=lambda a : (bool(a[2]), a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups_with_noise if g[3] > 1e-6]
                elif self.config.use_negative == "init2":
                    ent_groups = sorted(ent_groups, key=lambda a : (a[2], a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups if g[3] > 1e-6]
                else:
                    ent_groups = sorted(ent_groups, key=lambda a : (a[2], a[3]), reverse=True)
                    ent_groups = [g[:3] for g in ent_groups]

            else:
                ent_groups = sorted(ent_groups, key=lambda a : a[2], reverse=True)

        return ent_groups

    def hard_filter(self):
        pass
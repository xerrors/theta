import itertools
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchcrf import CRF
import numpy as np

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate=0.1):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class REModel(pl.LightningModule):

    def __init__(self, config, rel_ids, ent_ids, lmhead):
        super().__init__()
        self.config = config
        self.rel_ids = rel_ids
        self.ent_ids = ent_ids
        if config.use_rel_cls == 'multi_classifier':
            self.classifier = MultiNonLinearClassifier(config.model.hidden_size, len(rel_ids))
        elif config.use_rel_cls == 'lmhead':
            self.lmhead = lmhead

        # rel_embeddings
        self.rel_embeddings = nn.Embedding(len(rel_ids), config.model.hidden_size)

        # ent length embeddings
        self.ent_len_embeddings = nn.Embedding(10, config.model.hidden_size)

        self.filter_entity_pair_net = MultiNonLinearClassifier(config.model.hidden_size * 2, 1)

        self.sub_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.obj_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)

        if self.config.use_entity_pair_filter == "bilinear" or self.config.use_entity_pair_filter == "bilinear_proj":
            self.bilinear = nn.Bilinear(config.model.hidden_size, config.model.hidden_size, 1)

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

    def filter_entity_pair(self, hidden_state, entities, triples=None):

        logits = []
        map_dict = {}

        for i in range(len(entities)):
            if len(entities[i]) == 0: continue

            for j, e in enumerate(entities[i]):
                # 记录实体 e 在此 batch i 的所有实体中的位置，后面需要用到表格索引的
                map_dict[(i, e[0])] = j


            ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in entities[i]])    # (ent_num, hidden_size)
            ent_num, hidden_size = ent_hs.shape

            if self.config.use_entity_pair_filter == "cat_and_cls":
                ent_hs_x = ent_hs.unsqueeze(1).repeat(1, ent_num, 1)    # (ent_num, ent_num, hidden_size)
                ent_hs_y = ent_hs.unsqueeze(0).repeat(ent_num, 1, 1)    # (ent_num, ent_num, hidden_size)
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1).view(-1, hidden_size * 2)    # (ent_num, ent_num, hidden_size * 2)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)   # (ent_num * ent_num,)

            elif self.config.use_entity_pair_filter == "proj_then_cat":
                ent_hs_x = self.sub_proj(ent_hs)
                ent_hs_y = self.obj_proj(ent_hs)
                ent_hs_x = ent_hs_x.unsqueeze(1).repeat(1, ent_num, 1)    # (ent_num, ent_num, hidden_size)
                ent_hs_y = ent_hs_y.unsqueeze(0).repeat(ent_num, 1, 1)    # (ent_num, ent_num, hidden_size)
                ent_hs_pair = torch.cat([ent_hs_x, ent_hs_y], dim=-1).view(-1, hidden_size * 2)    # (ent_num, ent_num, hidden_size * 2)
                ent_hs_pair = self.filter_entity_pair_net(ent_hs_pair).squeeze(-1)   # (ent_num * ent_num,)

            elif self.config.use_entity_pair_filter == "attention":
                ent_hs_x = self.sub_proj(ent_hs)
                ent_hs_y = self.obj_proj(ent_hs)
                ent_hs_pair = torch.matmul(ent_hs_x, ent_hs_y.transpose(0,1)) / (ent_hs_x.shape[0] ** 0.5)    # (ent_num, ent_num, hidden_size)
                ent_hs_pair = ent_hs_pair.view(-1, 1).squeeze(-1)    # (ent_num, ent_num, hidden_size * 2)

            elif self.config.use_entity_pair_filter == "bilinear_proj":
                ent_hs_x = ent_hs.unsqueeze(1).repeat(1, ent_num, 1).view(-1, hidden_size)
                ent_hs_y = ent_hs.unsqueeze(0).repeat(ent_num, 1, 1).view(-1, hidden_size)
                ent_hs_pair = self.bilinear(ent_hs_x, ent_hs_y).squeeze(-1)

            elif self.config.use_entity_pair_filter == "bilinear":
                ent_hs_x = self.sub_proj(ent_hs)
                ent_hs_y = self.obj_proj(ent_hs)
                ent_hs_x = ent_hs_x.unsqueeze(1).repeat(1, ent_num, 1).view(-1, hidden_size)
                ent_hs_y = ent_hs_y.unsqueeze(0).repeat(ent_num, 1, 1).view(-1, hidden_size)
                ent_hs_pair = self.bilinear(ent_hs_x, ent_hs_y).squeeze(-1)

            logits.append(ent_hs_pair)

        logits = torch.cat(logits, dim=0)    # (batch_size * ent_num * ent_num,)

        loss = None
        if triples is not None:
            # sub_s, sub_e, obj_s, obj_e, rel_id, sub_type, obj_type
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

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return logits, loss, map_dict

    def get_triples_label(self, triples, device, ent_groups):

        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if len(triples) == 0:
            return triple_labels

        for i, pair in enumerate(ent_groups):
            b = pair[0]
                # t: [sub_start, sub_end, obj_start, obj_end, rel, sub_type, obj_type]
            for t in triples[b]:
                # 找不到
                if t[-1] == -1:
                    break
                # 找到了
                if t[:4].tolist() == pair[1:5]:
                    triple_labels[i] = t[4] + 1
                    break

        return triple_labels

    def prepare(self, theta, batch, hidden_state, triples, entities):
        """Get hidden state for the 2nd stage: relation classification"""
        assert self.config.use_two_stage, "use_two_stage must be True"

        ent_ids = theta.ent_ids
        device = hidden_state.device
        input_ids, _, pos, _, _, _ = batch
        bsz, seq_len = input_ids.shape

        logits, filter_loss, map_dict = self.filter_entity_pair(hidden_state, entities, triples)
        logits = logits.sigmoid()

        max_len= 512
        ent_groups = []
        rel_input_ids = []
        rel_positional_ids = []
        rel_attention_mask = []

        cls_token = theta.tokenizer.cls_token_id
        sep_token = theta.tokenizer.sep_token_id
        pad_token = theta.tokenizer.pad_token_id
        mask_token = theta.tokenizer.mask_token_id

        for b, entity in enumerate(entities):

            sent_s, sent_e = pos[b, 0], pos[b, 1]
            sent_len = sent_e - sent_s

            ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
            pos_ids = [b for b in range(sent_len+2)]
            masks = [1 for b in range(sent_len+2)]

            # 先构建一个初始的实体组，然后按照置信度排序，能排多少是多少
            draft_ent_groups = []
            for sub_pos, obj_pos in itertools.permutations(entity, 2):
                i = map_dict[(b, sub_pos[0])]
                j = map_dict[(b, obj_pos[0])]
                index = self.convert_bij_to_index((b, i, j), entities)
                draft_ent_groups.append((sub_pos, obj_pos, logits[index].item()))
            draft_ent_groups = sorted(draft_ent_groups, key=lambda a : a[-1], reverse=True)

            marker_mask = 1
            for ent_pair in draft_ent_groups:
                sub_pos, obj_pos, score = ent_pair

                if len(ids) + 3 <= max_len and score > self.config.get("ent_pair_threshold", 0):
                    marker_mask += 1
                    sub_begin_tag_id = ent_ids[sub_pos[2]+1]
                    obj_begin_tag_id = ent_ids[obj_pos[2]+1]
                    ids += [sub_begin_tag_id, mask_token, obj_begin_tag_id]

                    sub_pos_id = sub_pos[0] - sent_s + 1
                    obj_pos_id = obj_pos[0] - sent_s + 1

                    # experiments show that this is better than sub_pos_id and (sub_pos_id + obj_pos_id) // 2
                    mask_pos_id = obj_pos_id

                    pos_ids += [sub_pos_id, mask_pos_id, obj_pos_id]
                    masks += [marker_mask] * 3

                    ent_groups.append([b, sub_pos[0], sub_pos[1], obj_pos[0], obj_pos[1], sub_pos[2], obj_pos[2]])

            rel_input_ids.append(torch.tensor(ids))
            rel_positional_ids.append(torch.tensor(pos_ids))
            rel_attention_mask.append(torch.tensor(masks)) # 不要放到 cuda 上

        rel_input_ids = nn.utils.rnn.pad_sequence(rel_input_ids, batch_first=True, padding_value=pad_token)
        rel_positional_ids = nn.utils.rnn.pad_sequence(rel_positional_ids, batch_first=True, padding_value=0)

        # 2D attention mask
        padding_length = rel_input_ids.shape[1]
        rel_attention_mask_matrix = torch.zeros([bsz, padding_length, padding_length])

        for b, m in enumerate(rel_attention_mask):
            cur_len = len(m)
            matrix = []
            # 这里的 m.tolist() 会比之前要好，在计算上面
            for from_mask in m.tolist():
                matrix_i = []
                for to_mask in m.tolist():
                    # 每组实体只能看到自己的标记和句子中的文本
                    if to_mask == 1 or from_mask == to_mask:
                        matrix_i.append(1)
                    else:
                        matrix_i.append(0)

                matrix.append(matrix_i)
            rel_attention_mask_matrix[b, :cur_len, :cur_len] = torch.tensor(matrix)

        rel_attention_mask = rel_attention_mask_matrix.clone()

        rel_input_ids = rel_input_ids.to(device)
        rel_positional_ids = rel_positional_ids.to(device)
        rel_attention_mask = rel_attention_mask.to(device)
        assert rel_positional_ids.max() <= 512 and rel_positional_ids.min() >= 0, "positional ids Fault"
        assert rel_input_ids.shape == rel_positional_ids.shape

        rel_hidden_states = []

        # 从 triples 中构建标签
        triple_labels = self.get_triples_label(triples, device, ent_groups)
        if len(ent_groups) != 0:
            # 2. 重新计算 hidden state
            plm_model = theta.plm_model_for_re if self.config.use_independent_plm else theta.plm_model
            plm_model = plm_model.cuda() # 不知道为什么是 CPU，可能是因为 debug mode
            outputs = plm_model(
                rel_input_ids,
                attention_mask=rel_attention_mask,
                position_ids=rel_positional_ids,
                output_hidden_states=True
                )
            rel_stage_hs = outputs.hidden_states[-1]

            # 找到 mask 的 Hidden State
            mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
            rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]] # copilot NewBee

            if self.config.use_tag_to_pred_rel:
                sub_hidden_state = rel_stage_hs[mask_pos[0], mask_pos[1]-1]
                obj_hidden_state = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
                rel_hidden_states = torch.cat([rel_hidden_states, sub_hidden_state, obj_hidden_state], dim=-1)

        assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)
        return ent_groups, rel_hidden_states, triple_labels, filter_loss

    def forward(self, hidden_output, pos=None, labels=None):

        if len(hidden_output) == 0:
            return [], None

        if self.config.use_rel_cls == 'multi_classifier':
            logits = self.classifier(hidden_output)
        elif self.config.use_rel_cls == 'lmhead':
            logits = self.lmhead(hidden_output)
            logits = logits[..., self.rel_ids] # python Ellipsis operator
        else:
            raise NotImplementedError("unknown relation classifier: {}".format(self.config.use_rel_cls))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            # Only keep active parts of the loss
            if pos is not None:
                bsz, seq_len, _ = hidden_output.shape
                new_logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                new_labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)

                new_logits = new_logits.view(-1, len(self.rel_ids))
                new_labels = labels.view(-1)
            else:
                new_logits = logits.view(-1, len(self.rel_ids))
                new_labels = labels.view(-1)

            loss = loss_fct(new_logits, new_labels)

        return logits, loss


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

        self.num_ent_type = len(ent_ids) // 2

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
                # 判断是否是 I 标签以及是否是连续的
                elif logits[b, i] > self.num_ent_type and start:
                    # # 放松一点，只要是连续的就合并
                    # if entity[-1][2] == logits[b, i] - 1 - self.num_ent_type:
                    #     entity[-1][1] = i + 1
                    # else:
                    #     start = False
                    entity[-1][1] = i + 1 # 左闭右开
                else:
                    start = False

            entities.append(entity)

        return entities

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(FeedForward, self).__init__()
        # check the validity of the parameters
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers
        else:
            assert len(hidden_dims) == num_layers

        if isinstance(activations, str) or callable(activations):
            activations = [activations] * num_layers
        else:
            assert len(activations) == num_layers

        if isinstance(dropout, float) or isinstance(dropout, int):
            dropout = [dropout] * num_layers
        else:
            assert len(dropout) == num_layers

        # create a list of linear layers
        self.linear_layers = nn.ModuleList()
        input_dims = [input_dim] + hidden_dims[:-1]
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            self.linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        # create a list of activation functions
        self.activations = []
        for activation in activations:
            if activation == "relu":
                self.activations.append(nn.ReLU())
            elif activation == "gelu":
                self.activations.append(nn.GELU())
            elif callable(activation):
                self.activations.append(activation)
            else:
                raise ValueError("Invalid activation function")

        # create a list of dropout layers
        self.dropout = nn.ModuleList()
        for value in dropout:
            self.dropout.append(nn.Dropout(p=value))

    def forward(self, x):
        # loop over the layers and apply them sequentially
        for layer, activation, dropout in zip(
                self.linear_layers, self.activations, self.dropout):
            x = dropout(activation(layer(x)))

        return x


class SpanModel(pl.LightningModule):
    """【deprecated】用于NER的模型

    用途：
        1. 加载与命名实体相关的模型模块
        2. 计算此部分的损失

    """
    def __init__(self, config, num_labels):
        """初始化

        Args:
            config: 配置文件
            num_labels: 标签的数量(BIO)
            entity_type_num: 实体类型的标签数量
        """
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.model.hidden_dropout_prob)

        if config.use_span == 'classifier':
            self.classifier = nn.Linear(config.model.hidden_size, self.num_labels)
        elif config.use_span == 'multi_classifier':
            self.classifier = MultiNonLinearClassifier(config.model.hidden_size, self.num_labels)
        else:
            raise NotImplementedError(f"config.use_span={config.use_span} is not implemented!")

    def forward(self, hidden_output, pos=None, labels=None):

        hidden_output = self.dropout(hidden_output)
        logits = self.classifier(hidden_output)

        loss = None
        bsz = logits.shape[0]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            # Only keep active parts of the loss
            if pos is not None:
                labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                labels = labels.view(-1)
                new_logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                new_logits = new_logits.view(-1, self.num_labels)
            else:
                new_logits = logits.view(-1, self.num_labels)
                labels = labels.view(-1)

            loss = loss_fct(new_logits, labels)

        return logits, loss

    def decode_entities(self, logits, pos=None):

        if logits.shape[-1] == self.num_labels:
            logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz = logits.shape[0]
        for b in range(bsz):
            entity = []
            # 1 表示 B，即实体的开始，2 表示 I，即实体的中间
            start = False
            sent_start, sent_end = pos[b,0], pos[b,1]
            for i in range(sent_start, sent_end):
                if logits[b, i] == 1:
                    start = True
                    entity.append([i, i+1])
                elif logits[b, i] == 2 and start:
                    entity[-1][1] = i+1
                else:
                    start = False

            entities.append(entity)

        return entities


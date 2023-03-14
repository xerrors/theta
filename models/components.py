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


class SpanModel(pl.LightningModule):
    """用于NER的模型

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

        if self.config.use_rel_mask:
            self.prepare = self.prepare_with_mask
        else:
            self.prepare = self.prepare_origin

    def prepare_origin(self, theta, batch, hidden_state, triples, entities):
        """Get hidden state for the 2nd stage: relation classification

        Use Config:
            use_two_stage: True / False (default: False), 是否使用两阶段训练
            use_independent_plm: True / False (default: False), 是否使用独立的预训练模型
            use_rel_cls: multi_classifier / lmhead (default: lmhead), 关系分类器的类型
            use_ner: multi_classifier / lmhead (default: lmhead), 命名实体识别器的类型
            use_ent_type_in_rel: True / False (default: False), 是否使用实体类型作为关系分类的输入

        Constraints:
            1. use_two_stage must be True if use_independent_plm is True

        """
        input_ids, attention_mask, pos, _, _, _ = batch
        bsz, seq_len, h = hidden_state.shape

        config = self.config
        ent_ids = self.ent_ids
        device = self.device
        ent_num = len(config.dataset.ents)

        assert not config.use_independent_plm or config.use_two_stage # 使用独立的预训练模型，必须使用两阶段训练
        if config.use_two_stage:
            # 1. 在原本的句子后面拼上实体的 tag
            rel_input_ids = torch.zeros_like(input_ids).fill_(config.model.pad_token_id)
            rel_attention_mask = torch.zeros_like(attention_mask)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
            for i in range(bsz):
                tag = []
                pos_id = []
                entity = entities[i]
                sent_s, sent_e = pos[i, 0], pos[i, 1]
                for e in entity: # e: [start, end, ent_id] e.g. [[3, 4, 6], [4, 6, 2], [6, 12, 6]]
                    tag.append(ent_ids[e[2]+1]) # B tag, +1 是因为 O 的 id 是 0
                    tag.extend([ent_ids[e[2]+ent_num+1]] * (e[1] - e[0] - 1)) # I tag
                    pos_id.extend(np.arange(e[0], e[1]).tolist())

                assert len(tag) + sent_e <= seq_len, "entity is too long."
                assert len(tag) == len(pos_id), "tag and pos_id should have the same length."

                tag_len = len(tag)
                # tokens_num = min(seq_len, rel_attention_mask[i].sum().item() + tag_len)
                # rel_input_ids[i, tokens_num-tag_len-1:tokens_num-1] = torch.tensor(tag, device=device)
                # rel_input_ids[i, tokens_num-1] = theta.tokenizer.sep_token_id
                # position_ids[i, tokens_num-tag_len-1:tokens_num-1] = torch.tensor(pos_id, device=device)
                # rel_attention_mask[i, :tokens_num] = 1

                sent_len = sent_e - sent_s
                tag = torch.tensor(tag, device=hidden_state.device)
                pos_id = torch.tensor(pos_id, device=hidden_state.device)
                pos_id = pos_id - sent_s + 1 # 0 为 cls

                rel_input_ids[i, 0] = theta.tokenizer.cls_token_id                   # CLS
                rel_input_ids[i, 1:sent_len+1] = input_ids[i, sent_s:sent_e]         # SENT
                rel_input_ids[i, sent_len+1:sent_len+1+tag_len] = tag                # TAG
                rel_input_ids[i, sent_len+1+tag_len] = theta.tokenizer.sep_token_id  # SEP
                rel_attention_mask[i, :sent_len+1+tag_len+1] = 1
                position_ids[i, sent_len+1:sent_len+1+tag_len] = pos_id

            # 2. 重新计算 hidden state
            plm_model = theta.plm_model_for_re if self.config.use_independent_plm else theta.plm_model
            outputs = plm_model(rel_input_ids, attention_mask=rel_attention_mask, position_ids=position_ids, output_hidden_states=True)
            rel_stage_hs = outputs.hidden_states[-1]

        else:
            rel_stage_hs = hidden_state

        ent_groups = []
        for i, entity in enumerate(entities):
            for ent_pair in itertools.permutations(entity, 2):
                sub_pos, end_pos = ent_pair
                ent_groups.append([i, sub_pos[0], sub_pos[1], end_pos[0], end_pos[1], sub_pos[2], end_pos[2]])

        rel_hidden_states = []
        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if len(ent_groups) != 0:
            # 构建 rel_stage_hs, ent [batch_idx, sub_start, sub_end, end_start, end_end, sub_type, end_type]
            for ent in ent_groups:
                sent_s = pos[ent[0], 0]
                sub_hidden_state = rel_stage_hs[ent[0], ent[1]-sent_s+1]
                obj_hidden_state = rel_stage_hs[ent[0], ent[3]-sent_s+1]

                rel_hidden_state = obj_hidden_state - sub_hidden_state
                rel_hidden_states.append(rel_hidden_state)

            rel_hidden_states = torch.stack(rel_hidden_states, dim=0)

            # 从 triples 中构建标签
            triple_labels = self.get_triples_label(triples, device, ent_groups)

        return ent_groups, rel_hidden_states, triple_labels

    def get_triples_label(self, triples, device, ent_groups):
        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)
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

    def prepare_with_mask(self, theta, batch, hidden_state, triples, entities):
        """Get hidden state for the 2nd stage: relation classification"""
        assert self.config.use_two_stage, "use_two_stage must be True"

        ent_ids = theta.ent_ids
        input_ids, attention_mask, pos, _, _, _ = batch
        bsz, seq_len = input_ids.shape

        # 先以最大尺寸构建
        rel_input_ids = torch.zeros(bsz, 512, dtype=torch.long, device=hidden_state.device).fill_(theta.tokenizer.pad_token_id)
        rel_attention_mask = torch.zeros(bsz, 512, dtype=torch.long, device=hidden_state.device)
        position_ids = torch.arange(512, dtype=torch.long, device=hidden_state.device).expand(bsz, 512)

        # 1. 构建 rel_input_ids, rel_attention_mask, position_ids
        ent_groups = []
        for i, entity in enumerate(entities):
            tag = []
            pos_id = []
            sent_s, sent_e = pos[i, 0], pos[i, 1]
            for ent_pair in itertools.permutations(entity, 2):
                sub_pos, obj_pos = ent_pair
                tag.extend([ent_ids[sub_pos[2]+1], theta.tokenizer.mask_token_id, ent_ids[obj_pos[2]+1]])
                pos_id.extend([sub_pos[0], sub_pos[0], obj_pos[0]])
                ent_groups.append([i, sub_pos[0], sub_pos[1], obj_pos[0], obj_pos[1], sub_pos[2], obj_pos[2]])

            max_tag_len = 512 - (sent_e - sent_s) - 2
            tag = tag[:max_tag_len]
            pos_id = pos_id[:max_tag_len]
            assert len(tag) == len(pos_id) and len(tag) + sent_e <= 512, f"len(tag)={len(tag)}, len(pos_id)={len(pos_id)}, pos[i, 1]={pos[i, 1]}"

            sent_len = sent_e - sent_s

            tag = torch.tensor(tag, device=hidden_state.device)
            tag_len = len(tag)
            pos_id = torch.tensor(pos_id, device=hidden_state.device)
            pos_id = pos_id - sent_s + 1 # 0 为 cls

            """TODO
            可以考虑分段进行，但是计算开销以及代码逻辑会变得复杂一些，先测试一下直接舍弃的效果
            如果效果已经正常，则可以考虑分段进行
            """

            rel_input_ids[i, 0] = theta.tokenizer.cls_token_id                   # CLS
            rel_input_ids[i, 1:sent_len+1] = input_ids[i, sent_s:sent_e]         # SENT
            rel_input_ids[i, sent_len+1:sent_len+1+tag_len] = tag                # TAG
            rel_input_ids[i, sent_len+1+tag_len] = theta.tokenizer.sep_token_id  # SEP
            rel_attention_mask[i, :sent_len+1+tag_len+1] = 1
            position_ids[i, sent_len+1:sent_len+1+tag_len] = pos_id

        # 裁剪，以最小尺寸构建
        max_len = rel_attention_mask.sum(dim=1).max()
        rel_input_ids = rel_input_ids[:, :max_len]
        rel_attention_mask = rel_attention_mask[:, :max_len]
        position_ids = position_ids[:, :max_len]

        rel_hidden_states = []
        triple_labels = torch.zeros(len(ent_groups), device=hidden_state.device, dtype=torch.long)
        if len(ent_groups) != 0:
            # 2. 重新计算 hidden state
            # TODO: like PURE, add attention_mask
            plm_model = theta.plm_model_for_re if self.config.use_independent_plm else theta.plm_model
            outputs = plm_model(rel_input_ids, attention_mask=rel_attention_mask, position_ids=position_ids, output_hidden_states=True)
            rel_stage_hs = outputs.hidden_states[-1]

            # 找到 mask 的 Hidden State
            mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
            rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]] # copilot NewBee

            triple_labels = self.get_triples_label(triples, hidden_state.device, ent_groups)

        return ent_groups, rel_hidden_states, triple_labels


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
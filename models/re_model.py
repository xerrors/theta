import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.components import MultiNonLinearClassifier
from models.functions import getPretrainedLMHead

from utils.funcs import cosine_ease_in_out_minmax
from utils.optimizers import calc_num_training_steps

class REModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()
        self.config = theta.config
        self.rel_ids = theta.rel_ids
        self.ent_ids = theta.ent_ids

        config = self.config

        if config.use_rel == 'lmhead':
            if config.use_two_plm:
                self.lmhead = getPretrainedLMHead(theta.plm_model_for_re, config.model)
            else:
                self.lmhead = getPretrainedLMHead(theta.plm_model, config.model)

        elif config.use_rel == 'linear':
            self.classifier = nn.Linear(config.model.hidden_size, len(self.rel_ids))

        else:
            self.classifier = MultiNonLinearClassifier(config.model.hidden_size, len(self.rel_ids))

        self.filter_entity_pair_net = MultiNonLinearClassifier(config.model.hidden_size * 2, 1)

        self.sub_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.obj_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)

        self.grt_count = 0
        self.hit_count = 0
        self.cur_epoch = 0
        self.rel_type_num = len(self.config.dataset.ents)


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

        return logits, loss, map_dict

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

    def get_triples_label(self, triples, device, ent_groups, mode, epoch):
        """Get the labels of triples.

        Args:
            return_hit_rate: whether to return the hit rate of the triples.
        """

        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if len(triples) == 0:
            return triple_labels

        for i, pair in enumerate(ent_groups):
            b = pair[0]
            # t: [sub_start, sub_end, obj_start, obj_end, rel, sub_type, obj_type]
            for t in triples[b]:
                if t[-1] == -1:
                    break

                if t[:4].tolist() == pair[1:5]:
                    triple_labels[i] = t[4] + 1
                    break

        gt_count = 0
        for triple in triples:
            for t in triple:
                if t[-1] == -1:
                    break
                gt_count += 1

        hit_count = (triple_labels != 0).sum().item()

        if mode == "train":

            if epoch != self.cur_epoch:
                self.cur_epoch = epoch
                self.grt_count = 0
                self.hit_count = 0

            self.grt_count += gt_count
            self.hit_count += hit_count

        return triple_labels

    def prepare(self, theta, batch, hidden_state, entities, mode):
        """Get hidden state for the 2nd stage: relation classification"""

        ent_ids = theta.ent_ids
        device = hidden_state.device
        input_ids, _, pos, triples, _, _ = batch
        bsz, seq_len = input_ids.shape

        # 暂时使用 calc_num_training_steps 来反复计算，如果后面正式效果可行，再优化这部分的代码
        if self.config.ent_pair_threshold:
            ratio = cosine_ease_in_out_minmax(theta.global_step, calc_num_training_steps(theta))
            cur_threshold = self.config.ent_pair_threshold * ratio
        else:
            cur_threshold = 0

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
            filter_labels = self.get_filter_label(entities, triples, logits, map_dict)
            for sub_pos, obj_pos in itertools.permutations(entity, 2):
                i = map_dict[(b, sub_pos[0])]
                j = map_dict[(b, obj_pos[0])]
                index = self.convert_bij_to_index((b, i, j), entities)
                score = logits[index].item() if self.config.ner_rate > 0 else filter_labels[index].item()
                draft_ent_groups.append((sub_pos, obj_pos, score))
            draft_ent_groups = sorted(draft_ent_groups, key=lambda a : a[-1], reverse=True)

            marker_mask = 1
            for ent_pair in draft_ent_groups:
                (sub_s, sub_e, sub_t), (obj_s, obj_e, obj_t), score = ent_pair

                if len(ids) + 5 <= max_len and score > cur_threshold:
                    marker_mask += 1
                    if self.config.use_rel_opt1 == "ent_tag":
                        ss_tid = ent_ids[sub_t + 1]
                        os_tid = ent_ids[obj_t + 1]
                        se_tid = ent_ids[sub_t + self.rel_type_num + 1]
                        oe_tid = ent_ids[obj_t + self.rel_type_num + 1]
                    elif self.config.use_rel_opt1 == "new_tag":
                        ss_tid = theta.tag_ids[sub_t]
                        os_tid = theta.tag_ids[obj_t]
                        se_tid = theta.tag_ids[sub_t + self.rel_type_num]
                        oe_tid = theta.tag_ids[obj_t + self.rel_type_num]
                    else:
                        raise NotImplementedError(self.config.use_rel_opt1)

                    ss_pid = sub_s - sent_s
                    os_pid = obj_s - sent_s
                    se_pid = sub_e - sent_s + 2
                    oe_pid = obj_e - sent_s + 2

                    if self.config.use_rel_cat == 'pure':
                        ids += [mask_token, ss_tid, os_tid, se_tid, oe_tid]
                        pos_ids += [os_pid, ss_pid, os_pid, se_pid, oe_pid]
                        masks += [marker_mask] * 5
                    else:
                        ids += [mask_token, ss_tid, os_tid]
                        pos_ids += [os_pid, ss_pid, os_pid]
                        masks += [marker_mask] * 3

                    ent_groups.append([b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t])

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
        triple_labels = self.get_triples_label(triples, device, ent_groups, mode=mode, epoch=theta.current_epoch)

        if len(ent_groups) != 0:
            # 2. 重新计算 hidden state
            plm_model = theta.plm_model_for_re if self.config.use_two_plm else theta.plm_model
            plm_model = plm_model.cuda() # 不知道为什么是 CPU，可能是因为 debug mode
            outputs = plm_model(
                        rel_input_ids,
                        attention_mask=rel_attention_mask,
                        position_ids=rel_positional_ids,
                        output_hidden_states=True)
            rel_stage_hs = outputs.hidden_states[-1]

            # 找到 mask 的 Hidden State
            mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
            rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]] # copilot NewBee

            if self.config.use_ent_pred_rel == "tag":
                sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
                obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2]
                rel_hidden_states += sub_tag_hs + obj_tag_hs

            elif self.config.use_ent_pred_rel == "embed":
                sub_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+1]
                obj_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+2]
                sub_hs = rel_stage_hs[mask_pos[0], sub_pos_ids]
                obj_hs = rel_stage_hs[mask_pos[0], obj_pos_ids]
                rel_hidden_states += sub_hs + obj_hs

        assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)
        return ent_groups, rel_hidden_states, triple_labels, filter_loss

    def forward(self, theta, batch, hidden_state, entities, return_loss, mode):
        output = self.prepare(
                        theta=theta,
                        batch=batch,
                        hidden_state=hidden_state,
                        entities=entities,
                        mode=mode)

        if mode == 'train' and self.grt_count != 0:
            theta.log("info/hit_rate", self.hit_count / self.grt_count)

        ent_groups, hidden_output, triple_labels, filter_loss = output

        if len(hidden_output) == 0:
            rel_loss = torch.tensor(0.0).to(hidden_state.device)
            return ([],) if not return_loss else ([], rel_loss, filter_loss)

        if theta.graph is not None:
            hidden_output = theta.graph.query_rels(hidden_output)

        if self.config.use_rel == 'lmhead':
            assert self.lmhead is not None
            logits = self.lmhead(hidden_output)
            logits = logits[..., self.rel_ids] # python Ellipsis operator # BUG
        else:
            logits = self.classifier(hidden_output)

        triples_pred = []
        relation_logits = logits.argmax(dim=-1)
        for i in range(len(ent_groups)):
            rel = relation_logits[i].item()
            triples_pred.append(ent_groups[i] + [rel])
            if relation_logits[i] > 0 and theta.graph:
                rel_embeddings = hidden_output[i].detach().clone()
                theta.graph.add_edge(sub=ent_groups[i][0], obj=ent_groups[i][1], rel_type=rel-1, embedding=rel_embeddings)

        # triples_pred = [ent_groups[i] + [rel] for i in range(len(ent_groups))]

        rel_loss = torch.tensor(0.0).to(logits.device)
        if triple_labels is not None and return_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            new_logits = logits.view(-1, len(self.rel_ids))
            new_labels = triple_labels.view(-1)

            rel_loss = loss_fct(new_logits, new_labels)

            return (triples_pred, rel_loss, filter_loss)

        return (triples_pred,)


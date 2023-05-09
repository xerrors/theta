import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.components import MultiNonLinearClassifier, SelfAttention
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
        hidden_size = config.model.hidden_size

        if config.use_rel == 'lmhead':
            if config.use_two_plm:
                self.lmhead = getPretrainedLMHead(theta.plm_model_for_re, config.model)
            else:
                self.lmhead = getPretrainedLMHead(theta.plm_model, config.model)

        elif config.use_rel == 'linear':
            self.classifier = nn.Linear(hidden_size, len(self.rel_ids))

        else:
            self.classifier = MultiNonLinearClassifier(hidden_size, len(self.rel_ids))

        self.grt_count = 0
        self.hit_count = 0
        self.rel_count = 0
        self.cur_epoch = 0
        self.pre_mode = None
        self.filter_score = {} # p, r, f1
        self.rel_type_num = len(self.config.dataset.ents)

    def get_triples_label(self, triples, device, ent_groups, mode, epoch):
        """Get the labels of triples.

        Args:
            return_hit_rate: whether to return the hit rate of the triples.
        """

        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if triples is not None and len(triples) == 0:
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
        rel_count = len(triple_labels)

        if mode != "train" and mode == self.pre_mode:
            self.grt_count += gt_count
            self.hit_count += hit_count
            self.rel_count += rel_count

        return triple_labels

    def prepare(self, theta, hidden_state, batch, entities, mode="train"):
        """Get hidden state for the 2nd stage: relation classification"""

        ent_ids = theta.ent_ids
        device = hidden_state.device

        # when in predict mode, only input_ids is not None, others are None
        input_ids, _, pos, triples, _, _, _ = batch

        bsz, seq_len = input_ids.shape

        # is_train_thres = self.config.ent_pair_threshold and self.config.use_thres_train and mode == 'train'
        # is_val_thres = self.config.ent_pair_threshold and self.config.use_thres_val and mode != 'train'
        # use_thres = is_train_thres or is_val_thres

        # 如果启用 use_thres_train，则训练阶段实际上是使用的是所有的有关系的实体对
        # 如果启用 use_thres_val，则验证阶段使用的都是置信度高的实体对
        # 暂时使用 calc_num_training_steps 来反复计算，如果后面正式效果可行，再优化这部分的代码
        cur_threshold = -1
        if self.config.use_thres_val and mode != 'train':
            assert self.config.ent_pair_threshold >= 0, "ent_pair_threshold must be >= 0"
            cur_threshold = self.config.ent_pair_threshold
        elif mode != 'train' and mode != "predict":
            cur_threshold = 0.001

        if mode != "predict":
            logits, filter_loss, map_dict = theta.filter(hidden_state, entities, triples, mode)
            labels = theta.filter.get_filter_label(entities, triples, logits, map_dict)
        else:
            logits, filter_loss, map_dict = theta.filter(hidden_state, entities, mode=mode)
            labels = None

        max_len= 1024
        ent_groups = []
        rel_input_ids = []
        rel_positional_ids = []
        rel_attention_mask = []

        cls_token = theta.tokenizer.cls_token_id
        sep_token = theta.tokenizer.sep_token_id
        pad_token = theta.tokenizer.pad_token_id
        mask_token = theta.tokenizer.mask_token_id

        # 获取实体的数量
        ent_num = 0
        for entity in entities:
            ent_num += len(entity)

        # # 训练的时候使用真实标签来训练，测试的时候使用预测标签来训练
        # if mode == "train" or self.config.use_gold_filter_val or self.config.filter_rate == 0:
        #     logits = labels

        for b, entity in enumerate(entities):

            if pos is not None:
                sent_s, sent_e = pos[b, 0], pos[b, 1]
                sent_len = sent_e - sent_s
            else:
                sent_len = (input_ids[b] != pad_token).sum().item()
                sent_s, sent_e = 0, sent_len

            ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
            pos_ids = [b for b in range(sent_len+2)]
            masks = [1 for b in range(sent_len+2)]

            # 先构建一个初始的实体组，然后按照置信度排序，能排多少是多少
            if mode != "predict":
                gold_draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, labels, mode)
                pred_draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)

                # 如果当前的 epoch 大于最大的 epochs 的 1/3 的时候，就开始使用阈值过滤
                count = len(gold_draft_ent_groups)
                if self.config.use_thres_train and mode == "train":
                    r = self.filter_score.get("recall", 0)
                    pred_count = len([1 for e in pred_draft_ent_groups if e[-1] > 0.5]) # use 0.1 as threshold TODO
                    # c = max(\theta, c - c * r + \hat{c} * r)
                    count = max(5, int(count - count * r + pred_count * 2 * r))

                if mode == "train" or self.config.use_gold_filter_val or self.config.filter_rate == 0:
                    draft_ent_groups = gold_draft_ent_groups[:count]
                else:
                    draft_ent_groups = pred_draft_ent_groups

                gold_count = len([1 for e in gold_draft_ent_groups if e[-1] != 0])
                assert gold_count == 0 or gold_count != len(gold_draft_ent_groups), "不对劲"
                assert gold_count == 0 or len(ids) + gold_count * 3 <= max_len, "实体过多，超过最大长度"

            else:
                draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)

            marker_mask = 1
            for ent_pair in draft_ent_groups:
                sub_s, sub_e, sub_t = ent_pair[0][:3]
                obj_s, obj_e, obj_t = ent_pair[1][:3]
                score = ent_pair[2]

                if len(ids) + 3 <= max_len and score > cur_threshold:  # 当设置 filter_rate 为 0 时，仅包含正确的实体对
                    marker_mask += 1
                    ss_tid = theta.tag_ids[sub_t]
                    os_tid = theta.tag_ids[obj_t]
                    se_tid = theta.tag_ids[sub_t + self.rel_type_num]
                    oe_tid = theta.tag_ids[obj_t + self.rel_type_num]
                    ss_pid = sub_s - sent_s
                    os_pid = obj_s - sent_s
                    se_pid = sub_e - sent_s + 2
                    oe_pid = obj_e - sent_s + 2

                    ids += [mask_token, ss_tid, os_tid]
                    pos_ids += [os_pid, ss_pid, os_pid]
                    masks += [marker_mask] * 3

                    ent_groups.append([b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t, score])

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
        assert rel_positional_ids.max() <= 512 and rel_positional_ids.min() >= 0, "positional ids error"
        assert rel_input_ids.shape == rel_positional_ids.shape

        rel_hidden_states = []

        if len(ent_groups) != 0:
            # 2. 重新计算 hidden state
            plm_model = theta.plm_model_for_re if self.config.use_two_plm else theta.plm_model
            outputs = plm_model(
                        rel_input_ids, # torch.Size([8, 607])
                        attention_mask=rel_attention_mask, # torch.Size([8, 607, 607])
                        position_ids=rel_positional_ids, # torch.Size([8, 607]),
                        token_type_ids = torch.zeros(rel_input_ids.shape[:2], dtype=torch.long, device=device),
                        output_hidden_states=True)
            rel_stage_hs = outputs.hidden_states[-1]

            # 找到 mask 的 Hidden State
            mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
            rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]] # copilot NewBee

            if self.config.use_ent_pred_rel == "tag":
                sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
                obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2]
                rel_hidden_states += sub_tag_hs - obj_tag_hs

            elif self.config.use_ent_pred_rel == "embed":
                sub_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+1]
                obj_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+2]
                sub_hs = rel_stage_hs[mask_pos[0], sub_pos_ids]
                obj_hs = rel_stage_hs[mask_pos[0], obj_pos_ids]
                rel_hidden_states += sub_hs - obj_hs

        if mode != 'predict':
            triple_labels = self.get_triples_label(triples, device, ent_groups, mode=mode, epoch=theta.current_epoch)
            assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)

        else:
            triple_labels = None

        return ent_groups, rel_hidden_states, triple_labels, filter_loss

    def forward(self, theta, batch, hidden_state, entities=None, return_loss=False, mode="train", with_score=False):

        # if self.config.use_rel_opt1 == "filter":
        #     prepare = self.prepare
        # elif self.config.use_rel_opt1 == "batch":
        #     prepare = self.prepare_one_sent
        # else:
        #     raise NotImplementedError("use_rel_opt1: {} not implemented".format(self.config.use_rel_opt1))

        # output = prepare(
        #             theta=theta,
        #             batch=batch,
        #             hidden_state=hidden_state,
        #             entities=entities,
        #             mode=mode)

        output = self.prepare(
                    theta=theta,
                    batch=batch,
                    hidden_state=hidden_state,
                    entities=entities,
                    mode=mode)


        '''
        这里的 hit_rate 是指过滤后，正确的候选实体对与真实实体对的数量之比
        所体现的是关系抽取和实体过滤工作作用下的结果
        最终的 F1 在 PR 相差不大的情况下，可以理解为 hit_rate * rel_f1
        '''
        if mode != 'train' and mode != 'predict' and (mode != self.pre_mode or mode == 'test'):
            self.filter_score["precision"] = self.hit_count / (self.rel_count + 1e-8)
            self.filter_score["recall"] = self.hit_count / (self.grt_count + 1e-8)
            self.filter_score["f1"] = 2 * self.hit_count / (self.grt_count + self.rel_count + 1e-8)
            theta.log("info/pair_precision", self.filter_score["precision"])
            theta.log("info/pair_recall", self.filter_score["recall"])
            theta.log("info/pair_f1", self.filter_score["f1"])
            self.hit_count = 0
            self.grt_count = 0
            self.rel_count = 0

        self.pre_mode = mode

        ent_groups, hidden_output, triple_labels, filter_loss = output

        if len(hidden_output) == 0:
            rel_loss = torch.tensor(0.0).to(hidden_state.device)
            return ([],) if not return_loss else ([], rel_loss, filter_loss)

        if theta.graph is not None and mode != 'predict':
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
            if with_score:
                triples_pred.append(ent_groups[i] + [rel, logits[i]])
            else:
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
        
        if mode == "predict":
            return (triples_pred, ent_groups)

        return (triples_pred,)


    # def prepare_one_sent(self, theta, batch, hidden_state, entities, mode):

    #     ent_ids = theta.ent_ids
    #     device = hidden_state.device
    #     input_ids, _, pos, triples, _, _, _ = batch
    #     bsz, seq_len = input_ids.shape

    #     ent_groups = []
    #     rel_hidden_states = []

    #     cls_token = theta.tokenizer.cls_token_id
    #     sep_token = theta.tokenizer.sep_token_id
    #     pad_token = theta.tokenizer.pad_token_id
    #     mask_token = theta.tokenizer.mask_token_id

    #     # 先构建一个初始的实体组，然后按照置信度排序，能排多少是多少
    #     logits, filter_loss, map_dict = theta.filter(hidden_state, entities, triples, mode)
    #     logits = logits.sigmoid()

    #     if mode == "train" or self.config.use_gold_filter_val or self.config.filter_rate == 0:
    #         logits = theta.filter.get_filter_label(entities, triples, logits, map_dict)

    #     plm_model = theta.plm_model_for_re if self.config.use_two_plm else theta.plm_model

    #     batches = []
    #     for b, entity in enumerate(entities):

    #         sent_s, sent_e = pos[b, 0], pos[b, 1]
    #         sent_len = sent_e - sent_s

    #         ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
    #         pos_ids = [b for b in range(sent_len+2)]
    #         masks = [1 for b in range(sent_len+2)]

    #         marker_mask = 1
    #         draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)
    #         for i, ent_pair in enumerate(draft_ent_groups):
    #             (sub_s, sub_e, sub_t), (obj_s, obj_e, obj_t), score = ent_pair

    #             ss_tid = theta.tag_ids[sub_t]
    #             os_tid = theta.tag_ids[obj_t]
    #             ss_pid = sub_s - sent_s
    #             os_pid = obj_s - sent_s

    #             if len(ids) + 3 > 512:
    #                 batches.append((ids, pos_ids, masks))
    #                 if len(batches) == 4:
    #                     mask_hs = self.calc_mask_hs(theta, plm_model, batches) # copilot NewBee
    #                     rel_hidden_states = torch.cat([rel_hidden_states, mask_hs], dim=0) if len(rel_hidden_states) > 0 else mask_hs
    #                     batches = []

    #                 ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
    #                 pos_ids = [b for b in range(sent_len+2)]
    #                 masks = [1 for b in range(sent_len+2)]
    #                 marker_mask = 1

    #             marker_mask += 1
    #             ids += [mask_token, ss_tid, os_tid]
    #             pos_ids += [os_pid, ss_pid, os_pid]
    #             masks += [marker_mask] * 3

    #             if i == len(draft_ent_groups) - 1:
    #                 batches.append((ids, pos_ids, masks))

    #             ent_g = [b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t]
    #             ent_groups.append(ent_g)

    #     if len(batches) != 0:
    #         mask_hs = self.calc_mask_hs(theta, plm_model, batches)
    #         rel_hidden_states = torch.cat([rel_hidden_states, mask_hs], dim=0) if len(rel_hidden_states) > 0 else mask_hs # type: ignore

    #     # 从 triples 中构建标签
    #     triple_labels = self.get_triples_label(triples, device, ent_groups, mode=mode, epoch=theta.current_epoch)

    #     assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)
    #     return ent_groups, rel_hidden_states, triple_labels, filter_loss


    # def calc_mask_hs(self, theta, plm_model, batches):
    #     max_len = max([len(b[0]) for b in batches])
    #     bid, bpos, bmasks = [], [], []
    #     rel_attention_mask_matrix = torch.zeros([len(batches), max_len, max_len])
    #     for b, (b_ids, b_pos_ids, b_masks) in enumerate(batches):
    #         b_ids += [theta.tokenizer.pad_token_id] * (max_len - len(b_ids))
    #         b_pos_ids += [0] * (max_len - len(b_pos_ids))
    #         cur_len = len(b_masks)
    #         matrix = []
    #         # 这里的 m.tolist() 会比之前要好，在计算上面
    #         for from_mask in b_masks:
    #             matrix_i = []
    #             for to_mask in b_masks:
    #                 # 每组实体只能看到自己的标记和句子中的文本
    #                 if to_mask == 1 or from_mask == to_mask:
    #                     matrix_i.append(1)
    #                 else:
    #                     matrix_i.append(0)

    #             matrix.append(matrix_i)
    #         rel_attention_mask_matrix[b, :cur_len, :cur_len] = torch.tensor(matrix)

    #         bid.append(b_ids)
    #         bpos.append(b_pos_ids)
    #         bmasks.append(b_masks)

    #     bid = torch.tensor(bid).to(theta.device)
    #     bpos = torch.tensor(bpos).to(theta.device)
    #     rel_attention_mask_matrix = rel_attention_mask_matrix.to(theta.device)
    #     # bmasks = torch.tensor(bmasks).to(theta.device)

    #     outputs = plm_model(
    #                     bid,
    #                     attention_mask=rel_attention_mask_matrix,
    #                     position_ids=bpos,
    #                     output_hidden_states=True)

    #     rel_stage_hs = outputs.hidden_states[-1]
    #     mask_pos = torch.where(bid == theta.tokenizer.mask_token_id)
    #     mask_hs = rel_stage_hs[mask_pos[0], mask_pos[1]]
    #     return mask_hs
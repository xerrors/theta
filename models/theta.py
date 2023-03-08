import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import itertools

from models.components import MultiNonLinearClassifier, SpanModel, REModel, EntityClassifier, EntityClassifierWithLMHead
from utils.metrics import f1_score
from utils.optimizers import get_optimizer, calc_num_training_steps


def getBertForMaskedLMClass(model_config):
    if model_config.model_type == "roberta":
        from transformers import RobertaForMaskedLM
        return RobertaForMaskedLM
    elif model_config.model_type == "bert":
        from transformers import BertForMaskedLM
        return BertForMaskedLM
    elif model_config.model_type == "albert":
        from transformers import AlbertForMaskedLM
        return AlbertForMaskedLM

def getPretrainedLMHead(model, model_config):
    """获取预训练语言模型的头部 hidden_size -> vocab_size"""
    if model_config.model_type == "roberta":
        return model.lm_head
    elif model_config.model_type == "bert":
        return model.cls.predictions
    elif model_config.model_type == "albert":
        return model.predictions


class Theta(pl.LightningModule):
    """https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule"""

    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.tokenizer = data.tokenizer

        ModelClass = getBertForMaskedLMClass(config.model)
        self.plm_model = ModelClass.from_pretrained(config.model.model_name_or_path)
        self.lmhead = getPretrainedLMHead(self.plm_model, config.model) # 预训练语言模型的头部，并不占用额外参数

        # 常用参数
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.hidden_size = config.model.hidden_size

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Others
        self.rel_num = len(config.dataset.rels)
        self.ent_num = len(config.dataset.ents)
        self.na_idx = data.rel2id.get(config.dataset.na_label, None)

        # 模型评估
        self.best_f1 = 0
        self.test_f1 = 0
        # self.eval_fn = partial(f1_score, rel_num=config.dataset.rel_num, na_idx=self.na_idx)
        self.extand_and_init_additional_tokens()
        self.register_components()

    def extand_and_init_additional_tokens(self):
        """扩展并初始化额外的 token"""
        config = self.config

        if self.na_idx is None:
            rels = config.dataset.rels + ['NA']
        else:
            rels = config.dataset.rels

        self.rel_tokens = [f"[R{i}]" for i in range(len(rels))]
        # ent_type_ids = [f"[E{i}]" for i in range(self.ent_num + 1)]
        # ent_base_ids = ["[SS]", "[SE]", "[OS]", "[OE]", "[ES]", "[EE]"]

        # 扩展词表
        special_tokens_dict = {'additional_special_tokens': self.rel_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.plm_model.resize_token_embeddings(len(self.tokenizer))

        self.rel_ids = self.tokenizer.convert_tokens_to_ids(self.rel_tokens)

        # 初始化关系的词向量
        word_embeddings = self.plm_model.get_input_embeddings().weight.data
        ace_rel_map = {
            'NA': 'no relation',
            'ART': 'artifact person',
            'ORG-AFF': 'organization affiliation',
            'GEN-AFF': 'general affiliation',
            'PHYS': 'physical location',
            'PER-SOC': 'personal social',
            'PART-WHOLE': 'part whole'
        }

        ace_ids = [self.tokenizer.encode(ace_rel_map[rel], add_special_tokens=False) for rel in rels]

        for i, rel_id in enumerate(self.rel_ids):
            word_embeddings[rel_id] = word_embeddings[ace_ids[i]].clone().mean(dim=-2)

    def register_components(self):
        """ 用于构建除了预训练语言模型之外的所有模型组件
        很多时候，使用哪些模型组件，使用哪些模块都是需要根据 config 来决定的
        """
        config = self.config

        if self.config.use_rel_maps:
            self.rel_embeddings = nn.Embedding(self.rel_num, self.hidden_size)
            self.sub_cls = MultiNonLinearClassifier(config.model.hidden_size * 2, 1)
            self.obj_cls = MultiNonLinearClassifier(config.model.hidden_size * 2, 1)

        if self.config.use_ent_corres:
            self.ent_corres = MultiNonLinearClassifier(config.model.hidden_size * 2, 1)

        if config.use_rel_cls:
            self.rel_model = REModel(self.config, self.rel_ids)

        if config.use_span:
            self.span_model = SpanModel(config, 3)

    def forward(self, batch, mode="train"):
        """Model forward

        Args:
            batch (tuple): batch data
            return_loss (bool, optional): return loss or not. Defaults to True.
        """

        input_ids, attention_mask, pos, triples, span_maps, ent_corres = batch

        # Forward
        outputs = self.plm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 一些参数
        hidden_state = outputs.hidden_states[-1]
        bsz, seq_len, h = hidden_state.shape

        output = {}
        output["hidden_state"] = hidden_state

        # [deprecated] 计算关系表
        if self.config.use_rel_maps:
            rel_tmp_idx = torch.arange(0, self.rel_num, device=input_ids.device)
            rel_tmp_idx = rel_tmp_idx.unsqueeze(0).repeat(bsz, 1) # [bsz, rel_num]
            rel_embedding = self.rel_embeddings(rel_tmp_idx) # [bsz, rel_num, h]
            rel_embedding = rel_embedding.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [bsz, seq_len, rel_num, h]

            seq_output = hidden_state.unsqueeze(2).expand(-1, -1, self.rel_num, -1)  # [bsz, seq_len, rel_num, h]
            seq_output = torch.cat([seq_output, rel_embedding], dim=-1) # [bsz, seq_len, rel_num, h * 2]

            sub_cls = self.sub_cls(seq_output).squeeze(-1) # [bsz, seq_len, rel_num]
            obj_cls = self.obj_cls(seq_output).squeeze(-1) # [bsz, seq_len, rel_num]

            rel_maps_logits = torch.cat([sub_cls, obj_cls], dim=-1) # [bsz, seq_len, rel_num * 2]
            output["rel_maps_logits"] = rel_maps_logits

        # [deprecated] 实体对应损失
        if self.config.use_ent_corres:
            ent_corres_logits = self.get_corres_tabel(hidden_state, hidden_state)
            output["ent_corres_logits"] = ent_corres_logits

        # [OPTIONAL] 命名实体识别损失
        if self.config.use_span:
            span_logits, span_loss = self.span_model(hidden_state, pos=pos, labels=span_maps)
            output["span_logits"] = span_logits

        entities = self.span_model.decode_entities(span_maps, pos=pos) # gold entities
        # entities = self.span_model.decode_entities(span_logits, pos=pos) # pred entities

        if self.config.use_rel_cls:
            assert self.config.use_span, "need to use NER model to get entity position."

            ent_groups, rel_hidden_states, triple_labels = self.rel_model.prepare(triples, hidden_state, entities)

            rel_logits, rel_loss = self.rel_model(rel_hidden_states, lmhead=self.lmhead, labels=triple_labels)
            triples_pred = [ent_groups[i] + [rel_logits[i].argmax().item()] for i in range(len(ent_groups))]

            output["triples_pred"] = triples_pred
            triples_gold_count = sum([len([t for t in t_ if t[-1] != -1]) for t_ in triples])
            self.log("pred_triples_rate", (len(triples_pred) + 1) / (triples_gold_count+1))

        # 计算损失
        if mode != "test":

            loss = torch.tensor(0.0, device=input_ids.device)

            # [deprecated] 关系表损失
            if self.config.use_rel_maps:
                rel_maps = None # TODO
                rel_mask = torch.zeros_like(rel_maps)
                for b in range(bsz):
                    rel_mask[b, pos[b, 0]:pos[b, 1]] = 1

                rel_maps_loss = self.bce_loss_fn(rel_maps_logits, rel_maps.float())
                rel_maps_loss = (rel_maps_loss * rel_mask).sum() / rel_mask.sum()
                self.log("rel_map_loss", rel_maps_loss)

            # [deprecated] 实体对应损失
            if self.config.use_ent_corres:
                ent_corres_mask = torch.zeros_like(ent_corres_logits)
                for b in range(bsz):
                    ent_corres_mask[b, pos[b, 0]:pos[b, 1], pos[b, 0]:pos[b, 1]] = 1

                ent_corres_loss = self.bce_loss_fn(ent_corres_logits, ent_corres.float())
                ent_corres_loss = (ent_corres_loss * ent_corres_mask).sum() / ent_corres_mask.sum()
                self.log("corres_loss", ent_corres_loss)
                loss += ent_corres_loss

            if self.config.use_span:
                self.log("span_loss", span_loss)
                loss += span_loss

            if self.config.use_rel_cls and rel_loss: # 有时是 None
                self.log("rel_loss", rel_loss)
                loss += rel_loss

            output["loss"] = loss

        return output


    def get_corres_tabel(self, sub_hs, obj_hs, rel_hs=None):
        """[deprecated] 获取对应的关系表

        Args:
            sub_hs (torch.Tensor): [bsz, seq_len, h]
            obj_hs (torch.Tensor): [bsz, seq_len, h]
            rel_hs (torch.Tensor, optional): [bsz, rel_num, h]. Defaults to None.

        Returns:
            torch.Tensor: [bsz, seq_len, seq_len]
        """

        # 使用跟 PRGC 类似的处理方法，cat
        sub_hs = sub_hs.unsqueeze(2).expand(-1, -1, obj_hs.shape[1], -1)
        obj_hs = obj_hs.unsqueeze(1).expand(-1, sub_hs.shape[2], -1, -1)

        # 生成对应关系表
        corres_table = self.ent_corres(torch.cat([sub_hs, obj_hs], dim=-1))
        corres_table = corres_table.squeeze(-1)

        return corres_table

    # Train https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.training_step

    def training_step(self, batch, batch_idx):

        loss = self(batch, mode="train")["loss"]
        self.log('train_loss', loss)

        lr_step = {}
        for i, pg in enumerate(self.optimizers().param_groups):
            lr_step[f"lr_{i}"] = pg["lr"]
        self.log_dict(lr_step)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, _, _, triples, _, _  = batch
        output = self(batch, mode="dev")

        pred_triples = set()
        for t in output["triples_pred"]:
            if t[5] != 0:
                sub_token = self.tokenizer.decode(input_ids[t[0], t[1]:t[2]])
                obj_token = self.tokenizer.decode(input_ids[t[0], t[3]:t[4]])
                pred_triples.add((sub_token, obj_token, self.config.dataset.rels[t[5]-1]))

        gold_triples = set()
        for b in range(input_ids.shape[0]):
            for t in triples[b]:
                if t[4] != -1:
                    sub_token = self.tokenizer.decode(input_ids[b, t[0]:t[1]])
                    obj_token = self.tokenizer.decode(input_ids[b, t[2]:t[3]])
                    gold_triples.add((sub_token, obj_token, self.config.dataset.rels[t[4]]))
                else:
                    break

        loss = output["loss"]
        self.log('val_loss', loss)

        return {
            'pred_triples': pred_triples,
            'gold_triples': gold_triples,
        }

    def validation_epoch_end(self, outputs):
        f1, p, r = f1_score(outputs)

        self.best_f1 = max(f1, self.best_f1)
        self.log_dict_values({'val_p': p, 'val_r': r})
        self.log_dict_values({'val_f1': f1, 'best_f1': self.best_f1}, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        input_ids, _, _, triples, _, _  = batch
        output = self(batch, mode="test")

        pred_triples = set()
        for t in output["triples_pred"]:
            if t[5] != 0:
                sub_token = self.tokenizer.decode(input_ids[t[0], t[1]:t[2]])
                obj_token = self.tokenizer.decode(input_ids[t[0], t[3]:t[4]])
                pred_triples.add((sub_token, obj_token, self.config.dataset.rels[t[5]-1]))

        gold_triples = set()
        for b in range(input_ids.shape[0]):
            for t in triples[b]:
                if t[4] != -1:
                    sub_token = self.tokenizer.decode(input_ids[b, t[0]:t[1]])
                    obj_token = self.tokenizer.decode(input_ids[b, t[2]:t[3]])
                    gold_triples.add((sub_token, obj_token, self.config.dataset.rels[t[4]]))
                else:
                    break

        return {
            'pred_triples': pred_triples,
            'gold_triples': gold_triples,
        }

    def test_epoch_end(self, outputs):

        f1, p, r = f1_score(outputs)

        self.test_f1 = f1
        self.log_dict_values({'test_f1': f1, 'test_p': p, 'test_r': r})

    # Optimizer https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def decode_label_triples(self, ent_corres, rel_maps, input_ids, pos):
        """Convert table to triplet, can be used for ground truth"""

        bsz, seq_len, rel_num = rel_maps.shape[0], rel_maps.shape[1], rel_maps.shape[2] // 2

        triples = set()
        for b in range(bsz):
            sent_start_token_idx, sent_end_token_idx = pos[b, 0], pos[b, 1]
            for rel in range(rel_num):
                sub_seq = rel_maps[b, :, rel]
                obj_seq = rel_maps[b, :, rel + rel_num]

                # 遍历实体序列，连续为 1 的序列为一个实体，并记录下来起始位置
                sub_entities = []  # 解码出的是闭区间
                for s_i in range(sent_start_token_idx, sent_end_token_idx):
                    if sub_seq[s_i] == 1:
                        if len(sub_entities) == 0 or sub_entities[-1][1] != s_i:
                            sub_entities.append([s_i, s_i])
                        else:
                            sub_entities[-1][1] = s_i

                obj_entities = []  # 闭区间
                for o_i in range(sent_start_token_idx, sent_end_token_idx):
                    if obj_seq[o_i] == 1:
                        if len(obj_entities) == 0 or obj_entities[-1][1] != o_i:
                            obj_entities.append([o_i, o_i])
                        else:
                            obj_entities[-1][1] = o_i

                for sub in sub_entities:
                    for obj in obj_entities:
                        if ent_corres[b, sub[0], obj[0]] == 1: # 2023/03/04 仅使用起始位置的分数
                            sub_token = self.tokenizer.decode(input_ids[b, sub[0]:sub[1]+1])
                            obj_token = self.tokenizer.decode(input_ids[b, obj[0]:obj[1]+1])
                            rel_name = self.config.dataset.rels[rel]
                            triples.add((sub_token, obj_token, rel_name))

        return triples

    def deocde_triples(self, ent_corres, rel_maps, input_ids, pos, with_confidence=False):
        """Convert table to triplet, can be used for both prediction and ground truth

        Args:
            ent_corres (torch.Tensor): [bsz, seq_len, seq_len]
            rel_maps (torch.Tensor): [bsz, seq_len, rel_num * 2]
            input_ids (torch.Tensor): [bsz, seq_len]
            pos (torch.Tensor): [bsz, 4] # sent_start, sent_end, sentence_ix, sentence_start_in_doc
            with_confidence (bool, optional): [description]. Defaults to False. Use confidence \
                to filter out low confidence triples.

        """
        ent_threshold = 0.5
        rel_threshold = 0.5

        bsz, seq_len, rel_num = rel_maps.shape[0], rel_maps.shape[1], rel_maps.shape[2] // 2

        triples = set()
        for b in range(bsz):
            sent_start_token_idx, sent_end_token_idx = pos[b, 0], pos[b, 1]
            for rel in range(rel_num):
                sub_seq = rel_maps[b, :, rel]
                obj_seq = rel_maps[b, :, rel + rel_num]

                # 遍历实体序列，连续为 1 的序列为一个实体，并记录下来起始位置
                sub_entities = []  # 解码出的是闭区间
                for s_i in range(sent_start_token_idx, sent_end_token_idx):
                    if sub_seq[s_i] > ent_threshold:
                        if len(sub_entities) == 0 or sub_entities[-1][1] != s_i:
                            sub_entities.append([s_i, s_i])
                        else:
                            sub_entities[-1][1] = s_i

                obj_entities = []  # 闭区间
                for o_i in range(sent_start_token_idx, sent_end_token_idx):
                    if obj_seq[o_i] > ent_threshold:
                        if len(obj_entities) == 0 or obj_entities[-1][1] != o_i:
                            obj_entities.append([o_i, o_i])
                        else:
                            obj_entities[-1][1] = o_i

                for sub in sub_entities:
                    for obj in obj_entities:
                        start_score = ent_corres[b, sub[0], obj[0]] # 2023/03/04 仅使用起始位置的分数

                        if with_confidence:
                            sub_token = self.tokenizer.decode(input_ids[b, sub[0]:sub[1]+1])
                            obj_token = self.tokenizer.decode(input_ids[b, obj[0]:obj[1]+1])
                            rel_name = self.config.dataset.rels[rel]
                            triples.add((sub_token, obj_token, rel_name, start_score.item()))

                        elif start_score > rel_threshold:
                            sub_token = self.tokenizer.decode(input_ids[b, sub[0]:sub[1]+1])
                            obj_token = self.tokenizer.decode(input_ids[b, obj[0]:obj[1]+1])
                            rel_name = self.config.dataset.rels[rel]
                            triples.add((sub_token, obj_token, rel_name))

        return triples

    def log_dict_values(self, d, **kwargs):
        for k, v in d.items():
            self.log(k, v, **kwargs)

    # def convert_table_to_triplet(self, table, input_ids, pos):
    #     """Convert table to triplet"""
    #     bsz, seq_len = table.shape[:2]

    #     visited = torch.zeros_like(table)
    #     triples = set()

    #     for b in range(bsz):
    #         table_b = table[b]
    #         for i in range(pos[b, 0], pos[b, 1]):
    #             for j in range(pos[b, 0], pos[b, 1]):

    #                 rel = table_b[i, j]
    #                 if i == j or rel == 0 or visited[b, i, j] == 1:
    #                     continue

    #                 k = i + 1
    #                 while k < pos[b, 1] and table_b[k, j] == rel:
    #                     k += 1

    #                 l = j + 1
    #                 while l < pos[b, 1] and table_b[i, l] == rel:
    #                     l += 1

    #                 visited[b, i:k, j:l] = 1

    #                 sub = self.tokenizer.decode(input_ids[b, i:k])
    #                 obj = self.tokenizer.decode(input_ids[b, j:l])
    #                 rel = self.config.dataset.rels[rel.item()-1]
    #                 triples.add((sub, obj, rel))

    #     return triples

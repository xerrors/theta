import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
# from torchcrf import CRF
import torch.nn.init as init
from models.components import MultiNonLinearClassifier, SelfAttention
from models.functions import getPretrainedLMHead

from transformers.models.bert.modeling_bert import BertAttention, BertOutput, BertIntermediate

from utils.Focal_Loss import focal_loss

# from utils.Focal_Loss import focal_loss

class EntAttentionLayer(nn.Module):

    def __init__(self, config, word_embeddings_fc, ent_ids):
        super().__init__()
        self.config = config
        self.word_embeddings_fc = word_embeddings_fc
        self.attention = BertAttention(config=self.config.model)
        self.crossattention = BertAttention(config=self.config.model)
        self.intermediate = BertIntermediate(config=self.config.model)
        self.output = BertOutput(config=self.config.model)

        self.ent_range = self.config.ent_attn_range
        self.ent_ids = ent_ids

        # 添加参数初始化
        # for name, param in self.named_parameters():
        #     if 'weight' in name and 'layer_norm' not in name and 'word_embeddings' not in name:
        #         init.xavier_normal_(param)
        #     elif 'bias' in name and 'word_embeddings' not in name:
        #         init.constant_(param, 0.0)

    def forward(self, hidden_states):
        tag_embeddings = self.word_embeddings_fc()(torch.tensor(self.ent_ids, device=hidden_states.device))
        tag_embeddings = tag_embeddings.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

        if self.ent_range > 0:
            attn_mask = torch.ones(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
            attn_mask = torch.tril(attn_mask, diagonal=self.ent_range) * torch.triu(attn_mask, diagonal=-self.ent_range) # 20
        else:
            attn_mask = None

        # attn_out = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attn_mask)[0]
        # hidden_states = self.layer_norm_1(hidden_states + attn_out)
        # attn_out = self.cross_attn(hidden_states, tag_embeddings, tag_embeddings)[0]
        # hidden_states = self.layer_norm_2(hidden_states + attn_out)
        # ffn_out = self.ffn(hidden_states)
        # hidden_states = self.layer_norm_3(hidden_states + ffn_out)

        attn_out = self.attention(hidden_states, attention_mask=attn_mask)[0]
        attn_out = self.crossattention(hidden_states=attn_out, encoder_hidden_states=tag_embeddings)[0]

        intermediate_output = self.intermediate(attn_out)
        hidden_states = self.output(intermediate_output, attn_out)

        return hidden_states


# class EntAttention(nn.Module):

#     def __init__(self, config, word_embeddings_fc, ent_ids) -> None:
#         super().__init__()
#         self.layers = nn.ModuleList([
#             EntAttentionLayer(config, word_embeddings_fc, ent_ids)
#             for _ in range(config.get("ent_attn_layer_num", 2))])

#     def forward(self, hidden_state):
#         for layer in self.layers:
#             hidden_state = layer(hidden_state)
#         return hidden_state


class EntDecoder(nn.Module):

    def __init__(self, config, theta):
        super().__init__()
        self.config = config
        self.ent_ids = theta.ent_ids

        hidden_size = config.model.hidden_size
        tag_size = len(self.ent_ids)

        self.ffn = MultiNonLinearClassifier(hidden_size, tag_size, layers_num=self.config.ent_mlp_layer_num) # 2
        self.ffn_bio = MultiNonLinearClassifier(hidden_size, 3)
        self.attn = nn.ModuleList([
            EntAttentionLayer(config, theta.plm_model.get_input_embeddings, self.ent_ids)
            for _ in range(config.ent_attn_layer_num)]) # 3

        # if self.config.use_ner == "embed":
        #     self.get_bio_tag_embedding = lambda d:theta.plm_model.get_input_embeddings()(torch.tensor(self.ent_ids, device=d))

    def forward(self, hidden_states):
        """
        logits, [logits_out], [attention_out]
        """

        output = {
            "logits": self.ffn(hidden_states),
            "logits_out": [self.ffn(hidden_states)], # 第一层的输出
            "attention_out": [hidden_states],
        }

        for li, layer in enumerate(self.attn):
            hidden_states = layer(hidden_states)
            logits = self.ffn(hidden_states)

            # if li == len(self.attn) - 1:
            #     # if self.config.use_ner == "embed":
            #     #     logits = torch.matmul(hidden_states, self.get_bio_tag_embedding(hidden_states.device).t())
            #     # else:
            #     logits = self.ffn(hidden_states)
            # else:
            #     if self.config.use_ner_layer_loss == "Bio":
            #         logits = self.ffn_bio(hidden_states)
            #     else:
            #         logits = self.ffn(hidden_states)

            output["logits"] = logits
            output["logits_out"].append(logits)
            output["attention_out"].append(hidden_states)

        return output



class NERModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()

        self.decoder = EntDecoder(theta.config, theta)

        self.config = theta.config
        self.ent_ids = theta.ent_ids
        self.ent_tags_count = len(self.ent_ids)
        # self.plm = theta.plm_model

        # config = self.config
        # hidden_size = config.model.hidden_size

        # if config.use_ner == "lmhead":
        #     self.lmhead = getPretrainedLMHead(theta.plm_model, config.model)

        # elif config.use_ner == "mlp":
        #     self.classifier = MultiNonLinearClassifier(hidden_size, self.ent_tags_count, layers_num=self.config.get("ent_mlp_layer_num", 2))

        # else:
        #     raise NotImplementedError(f"config.use_ner = {config.use_ner} is not implemented")

        # if self.config.use_crf:
        #     self.crf = CRF(self.ent_tags_count, batch_first=True)


        # if self.config.use_ent_tag_cross_attn == "local":
        #     from models.components import LocalAttention
        #     self.self_attn = LocalAttention(embed_size=hidden_size, window_size=self.config.ent_attn_range)
        # self.get_input_embeddings = theta.plm_model.get_input_embeddings
        # self.attn_layer = EntAttention(config, self.get_input_embeddings(), self.ent_ids)

        # self.self_attn = SelfAttention(embed_dim=hidden_size, range=self.config.ent_attn_range)
        # self.word_embeddings = theta.plm_model.get_input_embeddings()
        # self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, bias=True, batch_first=True)
        # self.layer_norm = nn.LayerNorm(hidden_size)

        self.num_ent_type = len(self.config.dataset.ents)
        # self.loss_weight = torch.FloatTensor([config.get("na_ner_weight", 1)] + [1] * self.num_ent_type * 2)

    def forward(self, hidden_states, labels=None, graph=None, mask=None, return_hs=False):

        if mask is None:
            mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)

        out = self.decoder(hidden_states)
        logits = out["logits"]

        if labels is None:
            return logits, torch.tensor(0.0, device=logits.device)

        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        new_logits = logits.view(-1, logits.shape[-1])[mask.reshape(-1) > 0]
        new_labels = labels.reshape(-1).long()[mask.reshape(-1) > 0]
        loss = loss_fct(new_logits, new_labels)

        if self.config.use_ner_focal_loss: # default: False
            loss_fct = focal_loss(num_classes=self.ent_tags_count)
            loss = loss_fct(new_logits, new_labels)

        if self.config.use_ner_layer_loss: # default: True
            loss = torch.tensor(0.0, device=logits.device)
            for lo_i, logits_out in enumerate(out["logits_out"][1:]):
                new_logits = logits_out.view(-1, logits_out.shape[-1])[mask.reshape(-1) > 0]

                if lo_i == len(out["logits_out"][1:]) - 1:
                    loss += loss_fct(new_logits, new_labels)
                else:
                    loss += loss_fct(new_logits, new_labels) * 0.1

        if return_hs:
            return logits, loss, out["attention_out"]
        else:
            return logits, loss

    def decode_entities(self, logits, pos=None, with_score=False, mask=None, mode="train"):
        """return 左闭右开 [[(start, end, type), (...)],[],[(...)]]"""

        ori_logits = logits.clone()
        if with_score:
            assert len(logits.shape) == 3 and logits.shape[-1] == self.ent_tags_count

        is_gt = len(logits.shape) != 3 or logits.shape[-1] != self.ent_tags_count
        if not is_gt:
            logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz, seq_len = logits.shape[0], logits.shape[1]
        # O B*7(1~8) I*7(8-15), self.num_ent_type = 7
        for b in range(bsz):
            entity = []

            if pos is not None:
                sent_start, sent_end = pos[b, 0], pos[b, 1]
            else:
                sent_start, sent_end = 1, seq_len - 1

            entity = self.default_decode_strategy(logits, with_score, ori_logits, b, entity, sent_start, sent_end)

            if self.config.use_wider_ent_decode and not is_gt and mode != "train":
                entity = self.hard_decode_strategy(logits, with_score, ori_logits, b, entity, sent_start, sent_end)
                entity = self.soft_decode_strategy(logits, with_score, ori_logits, b, entity, sent_start, sent_end)
                entity = self.remove_duplicate(entity)

            entities.append(entity)

        return entities

    def default_decode_strategy(self, logits, with_score, ori_logits, b, entity, sent_start, sent_end):
        start = False
        for i in range(sent_start, sent_end):
            # 判断是否是 B 标签
            if 0 < logits[b, i] <= self.num_ent_type:
                start = True
                ent_type_id = logits[b, i].item() - 1
                if with_score:
                    ent_score = ori_logits[b, i]
                    entity.append([i, i + 1, ent_type_id, ent_score])
                else:
                    entity.append([i, i + 1, ent_type_id])

            elif start and logits[b, i] > self.num_ent_type:
                entity[-1][1] = i + 1
            else:
                start = False

        return entity

    def hard_decode_strategy(self, logits, with_score, ori_logits, b, entity, sent_start, sent_end):
        start = False
        for i in range(sent_start, sent_end):
            if 0 < logits[b, i] <= self.num_ent_type:
                start = True
                ent_type_id = logits[b, i].item() - 1
                if with_score:
                    ent_score = ori_logits[b, i]
                    entity.append([i, i + 1, ent_type_id, ent_score])
                else:
                    entity.append([i, i + 1, ent_type_id])

            elif start and logits[b, i] == ent_type_id + self.num_ent_type + 1:
                entity[-1][1] = i + 1
            else:
                start = False

        return entity

    def soft_decode_strategy(self, logits, with_score, ori_logits, b, entity, sent_start, sent_end):
        start = False
        for i in range(sent_start, sent_end):
            if logits[b, i] > 0:
                ent_type_id = (logits[b, i].item() - 1) % self.num_ent_type

                if start and ent_type_id == entity[-1][2]:
                    entity[-1][1] = i + 1
                else:
                    start = True
                    if with_score:
                        ent_score = ori_logits[b, i]
                        entity.append([i, i + 1, ent_type_id, ent_score])
                    else:
                        entity.append([i, i + 1, ent_type_id])
            else:
                start = False

        return entity

    @staticmethod
    def remove_duplicate(entity):
        new_entity = []
        for ent in entity:
            if ent not in new_entity:
                new_entity.append(ent)
        return new_entity



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

    def __init__(self, config, tag_embeddings_fc, tag_size):
        super().__init__()
        self.config = config
        self.tag_embeddings_fc = tag_embeddings_fc
        # self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        # self.self_ouput = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Dropout(0.1))
        # self.self_norm = nn.LayerNorm(embed_dim)

        # self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        # self.cross_output = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Dropout(0.1))
        # self.cross_norm = nn.LayerNorm(embed_dim)

        # #     nn.Linear(embed_dim, embed_dim),
        # #     nn.GELU(),
        # #     nn.Dropout(0.1),
        # #     nn.Linear(embed_dim, embed_dim),
        # # )
        # self.layer_norm_1 = nn.LayerNorm(embed_dim)
        # self.layer_norm_2 = nn.LayerNorm(embed_dim)
        # self.layer_norm_3 = nn.LayerNorm(embed_dim)
        self.attention = BertAttention(config=self.config.model)
        self.crossattention = BertAttention(config=self.config.model)
        self.intermediate = BertIntermediate(config=self.config.model)
        self.output = BertOutput(config=self.config.model)

        self.ent_range = self.config.ent_attn_range
        self.tag_size = tag_size

        # 添加参数初始化
        # for name, param in self.named_parameters():
        #     if 'weight' in name and 'layer_norm' not in name and 'word_embeddings' not in name:
        #         init.xavier_normal_(param)
        #     elif 'bias' in name and 'word_embeddings' not in name:
        #         init.constant_(param, 0.0)

    def forward(self, hidden_states):
        tag_embeddings = self.tag_embeddings_fc([i for i in range(self.tag_size)])
        tag_embeddings = tag_embeddings.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

        if self.ent_range:
            attn_mask = torch.ones(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
            attn_mask = torch.tril(attn_mask, diagonal=self.ent_range) * torch.triu(attn_mask, diagonal=-self.ent_range)
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

        hidden_size = config.model.hidden_size
        tag_size = len(theta.ent_ids)

        self.ffn = MultiNonLinearClassifier(hidden_size, tag_size, layers_num=self.config.ent_mlp_layer_num)
        self.ffn_bio = MultiNonLinearClassifier(hidden_size, 3)
        self.attn = nn.ModuleList([
            EntAttentionLayer(config, theta.get_tag_embeddings, tag_size)
            for _ in range(config.ent_attn_layer_num)])

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
            # logits = self.ffn(hidden_states)

            if li == len(self.attn) - 1:
                logits = self.ffn(hidden_states)
            else:
                if self.config.use_ner_layer_loss == "Bio":
                    logits = self.ffn_bio(hidden_states)
                else:
                    logits = self.ffn(hidden_states)

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
        # if self.config.use_ent_attn and not self.config.use_ent_tag_cross_attn:
        #     hidden_states = self.self_attn(hidden_states)

        # elif self.config.use_ent_tag_cross_attn:
        #     # assert (self.word_embeddings.weight == self.plm.get_input_embeddings().weight).all()
        #     # assert (self.plm.get_input_embeddings().weight == self.plm.get_output_embeddings().weight).all()
        #     tag_embeddings = self.word_embeddings(torch.tensor(self.ent_ids, device=hidden_states.device))
        #     tag_embeddings = tag_embeddings.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

        #     hidden_states = self.self_attn(hidden_states)
        #     attn_out = self.cross_attn(hidden_states, tag_embeddings, tag_embeddings)[0]
        #     hidden_states = self.layer_norm(hidden_states + attn_out)

        # if self.config.use_ent_bio_input:
        #     embeddings = self.get_input_embeddings()
        #     tag_embeddings = embeddings(torch.tensor(self.ent_ids, device=hidden_states.device)).mean(dim=0)
        #     hidden_states += tag_embeddings.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

        # if graph is not None:
        #     hidden_states = graph.query_ents(hidden_states)
        # hidden_states = self.attn_layer(hidden_states)

        # tag_embeddings = self.word_embeddings(torch.tensor(self.ent_ids, device=hidden_states.device))
        # tag_embeddings = tag_embeddings.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

        # hidden_states = self.self_attn(hidden_states)
        # attn_out = self.cross_attn(hidden_states, tag_embeddings, tag_embeddings)[0]
        # hidden_states = self.layer_norm(hidden_states + attn_out)

        if mask is None:
            mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)

        out = self.decoder(hidden_states)
        logits = out["logits"]

        # if self.config.use_ner == "lmhead":
        #     assert self.lmhead is not None
        #     logits = self.lmhead(hidden_states)
        #     logits = logits[..., self.ent_ids]
        # else:
        #     logits = self.classifier(hidden_states)

        # loss = torch.tensor(0.0, device=logits.device)
        # bsz = logits.shape[0]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            new_logits = logits.view(-1, logits.shape[-1])[mask.reshape(-1) > 0]
            new_labels = labels.reshape(-1).long()[mask.reshape(-1) > 0]
            loss = loss_fct(new_logits, new_labels)

            if self.config.use_ner_focal_loss:
                loss_fct = focal_loss(num_classes=self.ent_tags_count)
                loss = loss_fct(new_logits, new_labels)

            if self.config.use_ner_layer_loss and len(out["logits_out"]) > 1:
                loss = torch.tensor(0.0, device=logits.device)
                for lo_i, logits_out in enumerate(out["logits_out"][1:]):
                    new_logits = logits_out.view(-1, logits_out.shape[-1])[mask.reshape(-1) > 0]

                    if lo_i == len(out["logits_out"][1:]) - 1:
                        loss += loss_fct(new_logits, new_labels)
                    elif self.config.use_ner_layer_loss == "Bio":
                        bio_labels = new_labels.clone()
                        bio_labels[bio_labels > self.num_ent_type] = -1 # I -> -1
                        bio_labels[bio_labels > 0] = 1 # B -> 1
                        bio_labels[bio_labels == -1] = 2 # I -> 2
                        loss += loss_fct(new_logits, bio_labels) * (self.config.ner_layer_weight or 0.1)
                    else:
                        loss += loss_fct(new_logits, new_labels) * (self.config.ner_layer_weight or 0.1)
                
                if self.config.ner_layer_weight is not None:
                    loss = loss / (self.config.ner_layer_weight * len(out["logits_out"][1:]) + 1)

            # if self.config.use_crf:
            #     new_logits = nn.utils.rnn.pad_sequence([logits[i][mask[i] == 1] for i in range(bsz)], batch_first=True).cuda()
            #     labels = nn.utils.rnn.pad_sequence([labels[i][mask[i] == 1] for i in range(bsz)], batch_first=True).long().cuda()
            #     mask = nn.utils.rnn.pad_sequence([mask[i][mask[i] == 1] for i in range(bsz)], batch_first=True).bool().cuda()
            #     loss = -self.crf(new_logits, labels, mask=mask, reduction="token_mean")

            # else:
            #     loss_fct = nn.CrossEntropyLoss(reduction="mean", weight=self.loss_weight.cuda())
            #     new_logits = logits.view(-1, self.ent_tags_count)[mask.view(-1) > 0]
            #     new_labels = labels.view(-1).long()[mask.view(-1) > 0]

            #     loss = loss_fct(new_logits, new_labels)
            # if self.config.use_ner_focal_loss:
            #     loss_fct = focal_loss(num_classes=self.ent_tags_count)
            # else:
            #     loss_fct = nn.CrossEntropyLoss(reduction="mean", weight=self.loss_weight.cuda())

            # loss = loss_fct(new_logits, new_labels)

        if return_hs:
            return logits, loss, out["attention_out"]
        else:
            return logits, loss

    def decode_entities(self, logits, pos=None, with_score=False, mask=None):
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
            start = False

            if pos is not None:
                sent_start, sent_end = pos[b, 0], pos[b, 1]
            else:
                sent_start, sent_end = 1, seq_len - 1

            # if not is_gt and self.config.use_crf:
            #     sent_start, sent_end = 0, pos[b, 1] - pos[b, 0]

            for i in range(sent_start, sent_end):
                # 判断是否是 B 标签
                if logits[b, i] > 0 and logits[b, i] <= self.num_ent_type:
                    start = True
                    ent_type_id = logits[b, i].item() - 1
                    if with_score:
                        ent_score = ori_logits[b, i]
                        entity.append([i, i + 1, ent_type_id, ent_score])
                    else:
                        entity.append([i, i + 1, ent_type_id])

                # 判断是否是 I 标签
                # elif start and self.config.use_crf and logits[b, i] == (entity[-1][0] + self.num_ent_type):
                #     entity[-1][1] = i + 1  # 左闭右开
                # elif start and not self.config.use_crf and logits[b, i] > self.num_ent_type:
                #     entity[-1][1] = i + 1
                elif start and logits[b, i] > self.num_ent_type:
                    entity[-1][1] = i + 1
                else:
                    start = False

            # if self.config.use_crf and not is_gt:
            #     for ent in entity:
            #         ent[0] += pos[b, 0]
            #         ent[1] += pos[b, 0]

            entities.append(entity)

        return entities


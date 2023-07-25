import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchcrf import CRF

from models.components import MultiNonLinearClassifier, SelfAttention
from models.functions import getPretrainedLMHead


class NERModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()
        self.config = theta.config
        self.ent_ids = theta.ent_ids
        self.ent_tags_count = len(self.ent_ids)
        # self.plm = theta.plm_model

        config = self.config
        hidden_size = config.model.hidden_size

        if config.use_ner == "lmhead":
            self.lmhead = getPretrainedLMHead(theta.plm_model, config.model)

        elif config.use_ner == "linear":
            self.classifier = nn.Linear(hidden_size, self.ent_tags_count)

        elif config.use_ner == "mlp":
            if config.use_ent_tag_cross_attn:
                self.word_embeddings = theta.plm_model.get_input_embeddings()
                self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, bias=True, batch_first=True)
                self.layer_norm = nn.LayerNorm(hidden_size)
            self.classifier = MultiNonLinearClassifier(hidden_size, self.ent_tags_count, layers_num=2)

        else:
            raise NotImplementedError(f"config.use_ner = {config.use_ner} is not implemented")

        if self.config.use_crf:
            self.crf = CRF(self.ent_tags_count, batch_first=True)

        self.self_attn = SelfAttention(embed_dim=hidden_size, use_mask=True, range=self.config.ent_attn_range)

        self.num_ent_type = len(self.config.dataset.ents)
        self.loss_weight = torch.FloatTensor([config.get("na_ner_weight", 1)] + [1] * self.num_ent_type * 2)

    def forward(self, hidden_state, labels=None, graph=None, mask=None, return_hs=False):
        if self.config.use_ent_attn and not self.config.use_ent_tag_cross_attn:
            hidden_state = self.self_attn(hidden_state)

        elif self.config.use_ent_tag_cross_attn:
            # assert (self.word_embeddings.weight == self.plm.get_input_embeddings().weight).all()
            # assert (self.plm.get_input_embeddings().weight == self.plm.get_output_embeddings().weight).all()
            tag_embeddings = self.word_embeddings(torch.tensor(self.ent_ids, device=hidden_state.device))
            tag_embeddings = tag_embeddings.unsqueeze(0).repeat(hidden_state.shape[0], 1, 1)

            hidden_state = self.self_attn(hidden_state)
            attn_out = self.cross_attn(hidden_state, tag_embeddings, tag_embeddings)[0]
            hidden_state = self.layer_norm(hidden_state + attn_out)

        if graph is not None:
            hidden_state = graph.query_ents(hidden_state)

        if mask is None:
            mask = torch.ones(hidden_state.shape[:2], device=hidden_state.device)

        if self.config.use_ner == "lmhead":
            assert self.lmhead is not None
            logits = self.lmhead(hidden_state)
            logits = logits[..., self.ent_ids]
        else:
            logits = self.classifier(hidden_state)

        loss = torch.tensor(0.0, device=logits.device)
        bsz = logits.shape[0]

        if labels is not None:
            if self.config.use_crf:
                new_logits = nn.utils.rnn.pad_sequence([logits[i][mask[i] == 1] for i in range(bsz)], batch_first=True).cuda()
                labels = nn.utils.rnn.pad_sequence([labels[i][mask[i] == 1] for i in range(bsz)], batch_first=True).long().cuda()
                mask = nn.utils.rnn.pad_sequence([mask[i][mask[i] == 1] for i in range(bsz)], batch_first=True).bool().cuda()
                loss = -self.crf(new_logits, labels, mask=mask, reduction="token_mean")

            else:
                loss_fct = nn.CrossEntropyLoss(reduction="mean", weight=self.loss_weight.cuda())
                new_logits = logits.view(-1, self.ent_tags_count)[mask.view(-1) > 0]
                new_labels = labels.view(-1).long()[mask.view(-1) > 0]

                loss = loss_fct(new_logits, new_labels)

        if return_hs:
            return logits, loss, hidden_state
        else:
            return logits, loss

    def decode_entities(self, logits, pos=None, with_score=False, mask=None):
        """return 左闭右开 [[(start, end, type), (...)],[],[(...)]]"""

        ori_logits = logits.clone()
        if with_score:
            assert len(logits.shape) == 3 and logits.shape[-1] == self.ent_tags_count

        is_gt = len(logits.shape) != 3 or logits.shape[-1] != self.ent_tags_count
        if not is_gt:
            if self.config.use_crf:
                if mask is not None:
                    logits = nn.utils.rnn.pad_sequence(
                        [logits[i][mask[i] == 1] for i in range(logits.shape[0])],
                        batch_first=True,
                    ).cuda()
                logits = torch.tensor(self.crf.decode(logits), device=logits.device)
            else:
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

            if not is_gt and self.config.use_crf:
                sent_start, sent_end = 0, pos[b, 1] - pos[b, 0]

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
                elif start and self.config.use_crf and logits[b, i] == (entity[-1][0] + self.num_ent_type):
                    entity[-1][1] = i + 1  # 左闭右开
                elif start and not self.config.use_crf and logits[b, i] > self.num_ent_type:
                    entity[-1][1] = i + 1
                else:
                    start = False

            if self.config.use_crf and not is_gt:
                for ent in entity:
                    ent[0] += pos[b, 0]
                    ent[1] += pos[b, 0]

            entities.append(entity)

        return entities


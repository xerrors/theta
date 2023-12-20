import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.components import MultiNonLinearClassifier

class PreREModel(pl.LightningModule):

    def __init__(self, config):
        super(PreREModel, self).__init__()
        self.config = config
        self.hidden_size = config.model.hidden_size
        self.rel_num = len(config.dataset.rels)

        if config.use_pre_opt2 == "cat":
            feature_size = self.hidden_size * 2
        elif config.use_pre_opt2 == "proj_cos":
            feature_size = self.hidden_size // 4
            self._state = nn.Linear(self.hidden_size, feature_size, bias=False)
            self._tag = nn.Linear(self.hidden_size, feature_size, bias=False)
        else:
            feature_size = self.hidden_size

        self.classifier = MultiNonLinearClassifier(feature_size, 1)
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def gen_labels_for_pre_rel(self, triples):

        labels = torch.zeros((len(triples), self.rel_num), dtype=torch.float32)
        for i, ts in enumerate(triples):
            for t in ts:
                if t[4] == -1:
                    break
                labels[i][t[4]] = 1.0

        labels = labels.to(triples.device)
        return labels

    def forward(self, hidden_state, mask, rel_tag_embeds, mode="train", triples=None):

        if self.config.use_pre_opt1 == "whole":
            hidden_state = hidden_state.mean(dim=1)
        elif self.config.use_pre_opt1 == "cls":
            hidden_state = hidden_state[:, 0]
        else:
            hidden_state = torch.stack([
                hidden_state[i][mask[i] == 1].mean(dim=0) for i in range(len(hidden_state))
            ])

        fusion_type = self.config.use_pre_opt2 or "add" # default to add

        logits = None
        if fusion_type == "proj_cos":
            state = self._state(hidden_state)
            tag = self._tag(rel_tag_embeds)
            # add batch norm
            state = state / state.norm(dim=-1, keepdim=True)
            tag = tag / tag.norm(dim=-1, keepdim=True)
            logits = (state @ tag.T).sigmoid()

        else:

            state = hidden_state.unsqueeze(1).repeat(1, self.rel_num, 1)
            tag = rel_tag_embeds.unsqueeze(0).repeat(len(hidden_state), 1, 1)
            assert state.shape == tag.shape

            if fusion_type == "mul":
                state = state * tag
            elif fusion_type == "add":
                state = state + tag
            elif fusion_type == "cat":
                state = torch.cat([state, tag], dim=-1)

            logits = self.classifier(state).unsqueeze(-1)

        if triples is not None:
            flat_logits = logits.view(-1, self.rel_num).view(-1)
            labels = self.gen_labels_for_pre_rel(triples).view(-1)
            loss = self.loss(flat_logits, labels)
        else:
            loss = None

        return loss, logits
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.components import MultiNonLinearClassifier

class SpanEntityModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()
        self.config = theta.config
        self.ents = theta.config.dataset.ents
        self.num_ent_type = len(self.ents)

        config = self.config
        hidden_size = config.model.hidden_size

        self.max_span_len = config.get("max_span_length", 10)
        self.width_embed_size = 150
        self.width_embeddings = nn.Embedding(self.max_span_len, self.width_embed_size)

        self.inter_size = hidden_size * 2 + self.width_embed_size
        self.classifier = MultiNonLinearClassifier(
                                    hidden_size=self.inter_size,
                                    tag_size=self.num_ent_type + 1,
                                    layers_num=2,
                                    hidden_dim=150,
                                    dropout_rate=0.2)

        self.dropout = nn.Dropout(0.1)

        self.loss_weight = torch.FloatTensor([config.get("na_ner_weight", 1)] + [1] * self.num_ent_type)

    def convert_span_mask_to_spans(self, span_mask):
        '''左闭右开'''
        mask = torch.where(span_mask > 0)
        spans = []
        for i, j in zip(mask[0], mask[1]):
            start = i.item()
            end = j.item() + 1
            length = end - start
            assert start < end
            spans.append((start, end, length))

        return torch.tensor(spans, device=span_mask.device)

    def forward(self, hidden_state, span_mask, labels, graph, pos=None):

        device = hidden_state.device
        bs, l, d = hidden_state.shape

        hidden_state = self.dropout(hidden_state)

        h = []
        t = []
        for b in range(bs):
            spans = self.convert_span_mask_to_spans(span_mask[b])
            span_start = hidden_state[b, spans[:,0]]
            span_end = hidden_state[b, spans[:,1] - 1] # 左闭右开
            span_width = self.width_embeddings(spans[:,2]-1)
            ent = torch.cat([span_start, span_end, span_width], dim=-1)
            h.append(ent)
            t.append(labels[b, spans[:,0], spans[:,1]])

        h = torch.cat(h, dim=0)
        t = torch.cat(t, dim=0)
        assert h.shape[0] == t.shape[0]

        if graph is not None and self.inter_size == self.config.model.hidden_size:
            h = graph.query_ents(h, with_batch=False)

        logits = self.classifier(h) # b * l * l * n
        loss = torch.tensor(0.0, device=device)
        if labels is not None:
            # 按照权重计算 loss
            loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=self.loss_weight.cuda())
            loss = loss_fn(logits, t.long())

        del h, t
        return logits, loss

    def decode_entities(self, logits, span_mask, pos):
        if len(logits.shape) == 2 and logits.shape[-1] == (self.num_ent_type+1):
            logits = torch.argmax(logits, dim=-1)

        bs = pos.shape[0]
        e = []
        for b in range(bs):
            spans = self.convert_span_mask_to_spans(span_mask[b])
            for i, j, _ in spans:
                assert i < j
                e.append((b, i.item(), j.item()))

        entities = [[] for _ in range(bs)]
        x = torch.where(logits != 0)[0].long()
        for i in x:
            ent = e[int(i.item())]
            label = logits[int(i.item())].item() - 1
            assert label >= 0 and label < self.num_ent_type
            entities[ent[0]].append(ent[1:] + (label,))

        return entities

    def decode_gold_entities(self, ent_maps, pos):

        bs = pos.shape[0]
        entities = [[] for _ in range(bs)]
        indices  = torch.where(ent_maps!=0)
        for b, i, j in zip(indices[0], indices[1], indices[2]):
            label = ent_maps[b,i,j].item() - 1
            entities[b].append((i.item(), j.item(), label))

        return entities

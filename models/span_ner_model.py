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

        self.max_span_len = 15
        self.width_embed_size = 150
        self.width_embeddings = nn.Embedding(self.max_span_len, self.width_embed_size)

        if config.use_spert == "mlp":
            self.inter_size = hidden_size + self.width_embed_size
            self.classifier = MultiNonLinearClassifier(self.inter_size, self.num_ent_type + 1)
        else:
            self.inter_size = hidden_size
            self.classifier = MultiNonLinearClassifier(self.inter_size, self.num_ent_type + 1)

    def forward(self, hidden_state, labels, graph, pos):

        device = hidden_state.device
        bs, l, d = hidden_state.shape

        h = []
        t = []
        e = []
        for b in range(bs):
            sent_start, sent_end = pos[b,0], pos[b,1]
            for i in range(sent_start, sent_end):
                for j in range(1, self.max_span_len):
                    # TODO 优化并行运行的效率
                    if i + j <= sent_end:

                        opt1 = self.config.use_spert_opt1
                        if opt1 == "sum":
                            words_embed = hidden_state[b, i:(i+j)].sum(dim=-1)
                        elif opt1 == "mean":
                            words_embed = hidden_state[b, i:(i+j)].mean(dim=-1)
                        elif opt1 == "edge":
                            words_embed = hidden_state[b, 1] + hidden_state[b, i+j]
                        else:
                            words_embed = hidden_state[b, i:(i+j)].max(dim=0)[0]

                        width_embed = self.width_embeddings(torch.tensor(j, device=device))
                        h.append(torch.cat([words_embed, width_embed], dim=-1))
                        t.append(labels[b,i,i+j])
                        e.append((b, i, i+j))

        h = torch.stack(h)
        t = torch.stack(t)

        if graph is not None and self.inter_size == self.config.model.hidden_size:
            h = graph.query_ents(h, with_batch=False)

        logits = self.classifier(h) # b * l * l * n
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(logits, t.long())

        del h, t, e
        return logits, loss

    def decode_entities(self, logits, pos):
        if len(logits.shape) == 2 and logits.shape[-1] == (self.num_ent_type+1):
            logits = torch.argmax(logits, dim=-1)

        bs = pos.shape[0]
        e = []
        for b in range(bs):
            sent_start, sent_end = pos[b,0], pos[b,1]
            for i in range(sent_start, sent_end):
                for j in range(1, self.max_span_len):
                    if i + j <= sent_end:
                        e.append((b, i, i+j))

        entities = [[] for _ in range(bs)]
        x = torch.where(logits != 0)[0].long()
        for i in x:
            ent = e[int(i.item())]
            label = logits[int(i.item())].item()-1
            entities[ent[0]].append(ent[1:] + (label,))

        return entities

    def decode_gold_entities(self, ent_maps, pos):

        bs = pos.shape[0]
        entities = [[] for _ in range(bs)]
        indices  = torch.where(ent_maps!=0)
        for b, i, j in zip(indices[0], indices[1], indices[2]):
            label = ent_maps[b,i,j].item() - 1
            entities[b].append((i, j, label))

        return entities

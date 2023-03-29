import torch
import torch.nn as nn
import pytorch_lightning as pl
import networkx as nx

GraphCrossAttnLayer = nn.MultiheadAttention

class GraphCrossAttn(pl.LightningModule):

    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphCrossAttnLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        for layer in self.layers:
            q = layer(q, k, v)[0] + q
        out = self.layer_norm(q)
        out = self.dropout(out)
        out = self.output(out)
        return out


class RuntimeGraph(pl.LightningModule):

    def __init__(self, theta) -> None:
        super().__init__()

        self.G = nx.DiGraph() # 这是一个有向图
        self.config = theta.config
        self.layers = theta.config.get("graph_layers", 1)
        self.ent_ids = theta.ent_ids
        self.rel_ids = theta.rel_ids
        self.hidden_size = self.config.model.hidden_size

        self._ent = nn.Embedding(len(self.ent_ids), self.hidden_size)
        self._rel = nn.Embedding(len(self.rel_ids), self.hidden_size)

        self.ent_attn = GraphCrossAttn(self.hidden_size, 4, self.layers)
        self.rel_attn = GraphCrossAttn(self.hidden_size, 4, self.layers)

    def query_ents(self, x):
        e = self.get_entity()
        e = e.unsqueeze(0).repeat(x.shape[0], 1, 1)
        out = self.ent_attn(x, e, e)
        return out

    def query_rels(self, x):
        e = self.get_relation()
        out = self.rel_attn(x, e, e)
        return out

    def add_nodes(self, **kwargs):
        node_id = self.__len__()
        self.G.add_node(node_id, **kwargs)
        return node_id

    def add_edges(self, sub, obj, **kwargs):
        edge_id = len(self.G.edges)
        self.G.add_edge(sub, obj, edge_id=edge_id, **kwargs)
        return edge_id

    def update(self):
        nodes = torch.stack([node["embedding"] for node in self.G.nodes.values()])

    def get_entity(self):
        ent_idx = torch.arange(len(self.ent_ids), device=self.device)
        embeddings = self._ent(ent_idx)
        return embeddings

    def get_relation(self):
        rel_idx = torch.arange(len(self.rel_ids), device=self.device)
        embeddings = self._rel(rel_idx)
        return embeddings

    def __len__(self):
        return len(self.G.nodes)

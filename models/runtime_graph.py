import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import networkx as nx
import numpy as np


class GraphCrossAttnLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        '''Forward pass of the layer. Use standard multihead attention and add a residual connection.'''
        out, attn = self.attn(q, k, v)
        out = self.dropout1(out)
        out1 = self.layer_norm(out + q)

        out2 = self.output(out1)
        out2 = self.dropout2(out2)
        out = self.layer_norm(out1 + out2)
        return out, attn

class GraphCrossAttn(pl.LightningModule):

    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.1, batch_first=True):
        super().__init__()
        self.layers = nn.ModuleList([GraphCrossAttnLayer(hidden_size, num_heads, dropout, batch_first) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v):
        for layer in self.layers:
            q, attn = layer(q, k, v)
        out = self.output(q)
        return out


class RuntimeGraph(pl.LightningModule):

    def __init__(self, theta) -> None:
        super().__init__()

        self.G = nx.DiGraph() # 这是一个有向图
        self.config = theta.config
        self.layers = int(theta.config.use_graph_layers)
        self.ent_ids = theta.ent_ids
        self.rel_ids = theta.rel_ids
        self.hidden_size = self.config.model.hidden_size

        self._ent = nn.Embedding(len(self.ent_ids), self.hidden_size)
        self._rel = nn.Embedding(len(self.rel_ids), self.hidden_size)

        self.ent_attn = GraphCrossAttn(self.hidden_size, 4, self.layers)
        self.rel_attn = GraphCrossAttn(self.hidden_size, 4, self.layers)

    def query_ents(self, x, with_batch=True):
        e = self.get_entity()
        if with_batch:
            e = e.unsqueeze(0).repeat(x.shape[0], 1, 1)
        out = self.ent_attn(x, e, e)
        return out

    def query_rels(self, x):
        e = self.get_relation()
        out = self.rel_attn(x, e, e)
        return out

    def add_node(self, **kwargs):
        '''Add a node to the graph.

        Args:
            name: str, the name of the node.
            embedding: torch.Tensor, the embedding of the node.
            ent_type: int, the type of the node.
            node_type: str, the type of the node. 'entity' or 'relation'.
        '''

        node_id = self.__len__()
        self.G.add_node(node_id, **kwargs)
        return node_id

    def add_edge(self, sub, obj, **kwargs):
        """Add a edge

        Args:
            sub: subject
            obj: object
            rel: rel type
            embedding: embedding
        """
        edge_id = len(self.G.edges)
        self.G.add_edge(sub, obj, edge_id=edge_id, **kwargs)
        return edge_id

    def update(self):
        nodes = torch.stack([node['embedding'] for node in self.G.nodes.values()])

    def get_entity(self):
        ent_idx = torch.arange(len(self.ent_ids), device=self.device)
        embeddings = self._ent(ent_idx)

        # 获取 Node 的 Embedding # BUG 目前存在空节点的情况
        nodes = [node['embedding'] for node in self.G.nodes.values() if node.get('node_type') == 'entity']
        if len(nodes) > 0:
            node_embeddings = torch.stack(nodes)
            embeddings = torch.cat([embeddings, node_embeddings], dim=0)

        return embeddings

    def get_relation(self):
        rel_idx = torch.arange(len(self.rel_ids), device=self.device)
        embeddings = self._rel(rel_idx)

        # 获取 Edge 的 Embedding
        edges = [edge['embedding'] for edge in self.G.edges.values() if edge.get('edge_type') == 'relation']
        if len(edges) > 0:
            edge_embeddings = torch.stack(edges)
            embeddings = torch.cat([embeddings, edge_embeddings], dim=0)

        return embeddings

    def aggregate(self, x):
        '''Aggregation Function, also known as Message Passing Function'''
        pass

    def combinate(self, x):
        '''Combination Function, also known as Node Update Function'''
        pass

    def readout(self, x):
        '''Readout Function, also known as Graph Output Function'''
        pass

    def reset(self):
        """删除所有节点，所有边"""
        self.G = nx.DiGraph()

    def __len__(self):
        return len(self.G.nodes)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        '''Dense version of GAT.

        Args:
            nfeat: int, the number of features. input dim.
            nhid: int, the number of hidden units. hidden dim.
            nclass: int, the number of classes. output dim.
            dropout: float, dropout rate.
            alpha: float, LeakyReLU angle of the negative slope.
            nheads: int, the number of head attentions.
        '''
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    '''
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

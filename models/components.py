import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, layers_num=1, hidden_dim=None, dropout_rate=0.1):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.activation = nn.ReLU()

        if hidden_dim is None:
            hidden_dim = hidden_size // 2

        input_dims = [hidden_size] + [hidden_dim] * (layers_num - 1)
        output_dims = [hidden_dim] * layers_num

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], output_dims[i]),
                self.activation,
                nn.Dropout(dropout_rate)
            ) for i in range(layers_num)
        ])
        self.classifier = nn.Linear(hidden_dim, tag_size)

        # 参数初始化
        # for layer in self.layers:
        #     for name, param in layer.named_parameters():
        #         if 'weight' in name:
        #             nn.init.xavier_normal_(param)
        #         elif 'bias' in name:
        #             nn.init.constant_(param, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x


class SelfAttention(nn.Module):
    """ Self attention Layer including mask and LayerNorm"""
    def __init__(self, embed_dim, num_heads=4, ent_range=None):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # self.mask = None
        self.ent_range = ent_range

    # def create_mask(self, input_shape, input_device):

    #     mask = torch.zeros(input_shape[1], input_shape[1], device=input_device)
    #     for i in range(input_shape[1]):
    #         # mask[i, max(0, i-10):i+10] = 1.0
    #         # mask[i, i-10:i+10] = 1.0
    #         mask[i, max(0, i-self.range):i+self.range] = 1.0
    #     # mask = mask.unsqueeze(0)
    #     self.mask = mask

    def forward(self, x):
        # 创建 mask 矩阵
        # if self.range:
        #     if self.mask is None or self.mask.shape[0] < x.shape[1]:
        #         self.create_mask(x.shape, x.device)
        #         attn_mask = self.mask
        #     elif self.mask.shape[0] >= x.shape[1]:
        #         attn_mask = self.mask[:x.shape[1], :x.shape[1]].clone()
        #     else:
        #         raise ValueError("Mask shape error.")
        # else:
        #     attn_mask = None

        if self.ent_range:
            attn_mask = torch.ones(x.shape[1], x.shape[1], device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=self.ent_range) * torch.triu(attn_mask, diagonal=-self.ent_range)
        else:
            attn_mask = None

        # 执行 self-attention 操作
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        x = self.layer_norm(x + attn_output)
        return x


# class LocalAttention(torch.nn.Module):
#     def __init__(self, embed_size, window_size):
#         super(LocalAttention, self).__init__()
#         self.window_size = window_size
#         self.layer_norm = nn.LayerNorm(embed_size)

#         # Query, Key, Value linear projections
#         self.query = torch.nn.Linear(embed_size, embed_size)
#         self.key = torch.nn.Linear(embed_size, embed_size)
#         self.value = torch.nn.Linear(embed_size, embed_size)

#     def forward(self, x):
#         """
#         x: [batch_size, seq_length, embed_size]
#         """
#         B, L, E = x.size()

#         queries = self.query(x)
#         keys = self.key(x)
#         values = self.value(x)

#         outputs = []
#         for i in range(L):
#             # Define the local window limits
#             start = max(0, i - self.window_size)
#             end = min(L, i + self.window_size + 1)

#             # Extract local chunks
#             local_queries = queries[:, i, :].unsqueeze(1)  # [B, 1, E]
#             local_keys = keys[:, start:end, :]  # [B, W, E]
#             local_values = values[:, start:end, :]  # [B, W, E]

#             # Local attention score
#             scores = torch.bmm(local_queries, local_keys.transpose(1, 2)) / E**0.5  # [B, 1, W]
#             attn_probs = torch.softmax(scores, dim=-1)  # [B, 1, W]

#             # Compute output
#             output = torch.bmm(attn_probs, local_values).squeeze(1)  # [B, E]
#             outputs.append(output)

#         attn_output = torch.stack(outputs, dim=1)  # [B, L, E]
#         x = self.layer_norm(x + attn_output)
#         return x



# class MultiNonLinearClassifier(nn.Module):
#     def __init__(self, hidden_size, tag_size, dropout_rate=0.1):
#         super(MultiNonLinearClassifier, self).__init__()
#         self.tag_size = tag_size
#         self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, input_features):
#         features_tmp = self.linear(input_features)
#         features_tmp = nn.ReLU()(features_tmp)
#         features_tmp = self.dropout(features_tmp)
#         features_output = self.hidden2tag(features_tmp)
#         return features_output

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(FeedForward, self).__init__()
        # check the validity of the parameters
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers
        else:
            assert len(hidden_dims) == num_layers

        if isinstance(activations, str) or callable(activations):
            activations = [activations] * num_layers
        else:
            assert len(activations) == num_layers

        if isinstance(dropout, float) or isinstance(dropout, int):
            dropout = [dropout] * num_layers
        else:
            assert len(dropout) == num_layers

        # create a list of linear layers
        self.linear_layers = nn.ModuleList()
        input_dims = [input_dim] + hidden_dims[:-1]
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            self.linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        # create a list of activation functions
        self.activations = []
        for activation in activations:
            if activation == "relu":
                self.activations.append(nn.ReLU())
            elif activation == "gelu":
                self.activations.append(nn.GELU())
            elif callable(activation):
                self.activations.append(activation)
            else:
                raise ValueError("Invalid activation function")

        # create a list of dropout layers
        self.dropout = nn.ModuleList()
        for value in dropout:
            self.dropout.append(nn.Dropout(p=value))

    def forward(self, x):
        # loop over the layers and apply them sequentially
        for layer, activation, dropout in zip(
                self.linear_layers, self.activations, self.dropout):
            x = dropout(activation(layer(x)))

        return x

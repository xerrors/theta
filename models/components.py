import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, layers_num=1, hidden_dim=None, dropout_rate=0.1):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if hidden_dim is None:
            hidden_dim = hidden_size // 2

        input_dims = [hidden_size] + [hidden_dim] * (layers_num - 1)
        output_dims = [hidden_dim] * layers_num

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], output_dims[i]),
                self.activation,
                self.dropout
            ) for i in range(layers_num)
        ])
        self.classifier = nn.Linear(hidden_dim, tag_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, bias=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + attn_output)
        return x



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

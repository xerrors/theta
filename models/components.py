import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate=0.1):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SpanModel(pl.LightningModule):
    """用于NER的模型

    用途：
        1. 加载与命名实体相关的模型模块
        2. 计算此部分的损失

    """
    def __init__(self, config, num_labels):
        """初始化

        Args:
            config: 配置文件
            num_labels: 标签的数量(BIO)
            entity_type_num: 实体类型的标签数量
        """
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # self.classifier = nn.Sequential(
        #     FeedForward(input_dim=config.hidden_size,
        #                 num_layers=2,
        #                 hidden_dims=150,
        #                 activations='relu',
        #                 dropout=config.hidden_dropout_prob),
        #     nn.Linear(150, num_labels)
        # )

    def forward(self, hidden_output, pos=None, labels=None):

        hidden_output = self.dropout(hidden_output)
        logits = self.classifier(hidden_output)

        loss = None
        bsz = logits.shape[0]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            # Only keep active parts of the loss
            if pos is not None:
                labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                labels = labels.view(-1)
                new_logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                new_logits = new_logits.view(-1, self.num_labels)
            else:
                new_logits = logits.view(-1, self.num_labels)
                labels = labels.view(-1)

            loss = loss_fct(new_logits, labels)

        return logits, loss

    def decode_entities(self, logits, pos=None):

        if logits.shape[-1] == self.num_labels:
            logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz = logits.shape[0]
        for b in range(bsz):
            entity = []
            # 1 表示 B，即实体的开始，2 表示 I，即实体的中间
            start = False
            sent_start, sent_end = pos[b,0], pos[b,1]
            for i in range(sent_start, sent_end):
                if logits[b, i] == 1:
                    start = True
                    entity.append([i, i+1])
                elif logits[b, i] == 2 and start:
                    entity[-1][1] = i+1
                else:
                    start = False

            entities.append(entity)

        return entities


class REModel(pl.LightningModule):

    def __init__(self, config, rel_ids):
        super().__init__()
        self.config = config
        self.rel_ids = rel_ids
        # self.lmhead = lmhead
        # self.classifier = MultiNonLinearClassifier(config.model.hidden_size, len(rel_ids))

    def forward(self, hidden_output, lmhead, pos=None, labels=None):
        logits = lmhead(hidden_output)
        logits = logits[..., self.rel_ids] # python Ellipsis operator

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            # Only keep active parts of the loss
            if pos is not None:
                bsz, seq_len, _ = hidden_output.shape
                logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)

            logits = logits.view(-1, len(self.rel_ids))
            labels = labels.view(-1)

            loss = loss_fct(logits, labels)

        return logits, loss


class EntityClassifier(pl.LightningModule):

    def __init__(self, config, ent_ids):
        super().__init__()
        self.config = config
        self.ent_ids = ent_ids

        max_span_length = config.get("max_span_length", 10)
        width_embedding_dim = config.get("width_embedding_dim", 150)
        head_hidden_dim = config.get("head_hidden_dim", 150)

        self.classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations='relu',
                        dropout=config.hidden_dropout_prob),
            nn.Linear(head_hidden_dim, len(ent_ids))
        )

        # 暂时还不使用
        self.width_embedding = nn.Embedding(max_span_length+1, width_embedding_dim)

    def forward(self, hidden_output, pos=None, labels=None):
        """
        :param hidden_output: [bsz, seq_len, hidden_size * 2 + width_embedding_dim]
        :param pos: [bsz, 2]
        :param labels: [bsz, seq_len]
        :return:
        """
        logits = self.classifier(hidden_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            # Only keep active parts of the loss
            if pos is not None:
                bsz, seq_len, _ = hidden_output.shape
                logits = torch.cat([logits[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)
                labels = torch.cat([labels[b, pos[b,0]:pos[b,1]] for b in range(bsz)], dim=0)

            logits = logits.view(-1, len(self.ent_ids))
            labels = labels.view(-1)

            loss = loss_fct(logits, labels)

        return logits, loss


class EntityClassifierWithLMHead(pl.LightningModule):

    def __init__(self, config, ent_ids):
        super().__init__()
        self.config = config
        self.ent_ids = ent_ids

    def forward(self, hidden_output, lmhead, labels=None):
        logits = lmhead(hidden_output)
        logits = logits[..., self.ent_ids]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            logits = logits.view(-1, len(self.ent_ids))
            labels = labels.view(-1)

            loss = loss_fct(logits, labels)

        return logits, loss



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

class CRFSpanModel(pl.LightningModule):

    def __init__(self, config, ent_ids):
        super().__init__()
        self.config = config
        self.ent_ids = ent_ids
        self.crf = CRF(len(ent_ids), batch_first=True)

    def forward(self, hidden_output, pos=None, labels=None):
        logits = self.crf.decode(hidden_output)

        loss = None
        if labels is not None:
            loss = self.crf(hidden_output, labels, mask=pos)

        return logits, loss

# 实现一个 crf
# class CRF(nn.Module):

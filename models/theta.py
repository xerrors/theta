import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from models.components import MultiNonLinearClassifier
from utils.metrics import f1_score
from utils.optimizers import get_optimizer

# from models.functions import convert_table_to_triplet

# from data.utils import extend_maps_to_one_hot


class Theta(pl.LightningModule):
    """https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule"""

    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.tokenizer = data.tokenizer

        self.model = AutoModel.from_pretrained(config.model.model_name_or_path)

        # 常用参数
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.hidden_size = config.model.hidden_size

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Others
        self.na_idx = data.rel2id.get(config.dataset.na_label, None)

        # 模型评估
        self.best_f1 = 0
        self.test_f1 = 0
        # self.eval_fn = partial(f1_score, rel_num=config.dataset.rel_num, na_idx=self.na_idx)

        self.register_components()

    def register_components(self):
        """ 用于构建除了预训练语言模型之外的所有模型组件
        很多时候，使用哪些模型组件，使用哪些模块都是需要根据 config 来决定的，
        如果放在 __init__ 中，会导致代码臃肿
        """
        config = self.config
        self.sub_cls = MultiNonLinearClassifier(
            config.model.hidden_size * 2, 1)
        self.obj_cls = MultiNonLinearClassifier(
            config.model.hidden_size * 2, 1)

        # 实体的映射表，用于计算实体的对应关系
        self.ent_corres = MultiNonLinearClassifier(
            config.model.hidden_size * 2, 1)

        # 关系的 embedding，暂时还没有用到，后面会利用模型的参数来初始化这个 embedding
        self.rel_embedding = nn.Embedding(
            config.dataset.rel_num, config.model.hidden_size)  # 跟 nn.Parameter 有什么区别？

        # self.init_rel_embedding_weight()

    def init_rel_embedding_weight(self):
        """初始化关系的 embedding"""
        # 每一个关系所对应的自然语言描述
        ace_relation_mapping = {
            'ART': 'artifact person',
            'ORG-AFF': 'organization affiliation',
            'GEN-AFF': 'general affiliation',
            'PHYS': 'physical location',
            'PER-SOC': 'personal social',
            'PART-WHOLE': 'part whole'
        }

        ace_relation_tokens = [self.tokenizer.tokenize(
            ace_relation_mapping[rel]) for rel in self.config.dataset.rels]

        # 获取预训练的模型的embedding
        model_embedding = self.model.embeddings.word_embeddings.weight

        # 初始化 self.rel_embedding
        for i, tokens in enumerate(ace_relation_tokens):
            for token in tokens:
                self.rel_embedding.weight.data[i] += model_embedding[self.tokenizer.vocab[token]]

        self.rel_embedding.weight.data = self.rel_embedding.weight.data / \
            len(ace_relation_tokens)  # 除以关系的数量

    def forward(self, batch, return_details=False):
        """Forward"""

        input_ids, attention_mask, ner_maps, rel_maps, ent_corres, pos = batch

        # Forward
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # 一些参数
        rel_num = rel_maps.shape[2] // 2
        bsz, seq_len, h = outputs.last_hidden_state.shape

        # 实体对应损失
        ent_corres_pred = self.get_corres_tabel(
            outputs.last_hidden_state, outputs.last_hidden_state)

        ent_corres_mask = torch.zeros_like(ent_corres_pred)
        for b in range(bsz):
            ent_corres_mask[b, pos[b, 0]:pos[b, 1], pos[b, 0]:pos[b, 1]] = 1

        ent_corres_loss = self.bce_loss_fn(ent_corres_pred, ent_corres.float())
        ent_corres_loss = (ent_corres_loss *
                           ent_corres_mask).sum() / ent_corres_mask.sum()

        # 计算关系表
        rel_tmp_idx = torch.arange(
            0, rel_num, device=input_ids.device).unsqueeze(0).repeat(bsz, 1)
        rel_embedding = self.rel_embedding(rel_tmp_idx)
        rel_embedding = rel_embedding.unsqueeze(
            1).expand(-1, seq_len, -1, -1)  # [bsz, seq_len, rel_num, h]

        seq_output = outputs.last_hidden_state.unsqueeze(
            2).expand(-1, -1, rel_num, -1)  # [bsz, seq_len, rel_num, h]
        # [bsz, seq_len, rel_num, h * 2]
        seq_output = torch.cat([seq_output, rel_embedding], dim=-1)
        # [bsz, seq_len, rel_num]
        sub_cls = self.sub_cls(seq_output).squeeze(-1)
        # [bsz, seq_len, rel_num]
        obj_cls = self.obj_cls(seq_output).squeeze(-1)

        # [bsz, seq_len, rel_num * 2]
        rel_maps_pred = torch.cat([sub_cls, obj_cls], dim=-1)

        rel_mask = torch.zeros_like(rel_maps, dtype=torch.float32)
        for b in range(bsz):
            rel_mask[b, pos[b, 0]:pos[b, 1]] = 1

        rel_map_loss = self.bce_loss_fn(rel_maps_pred, rel_maps.float())
        rel_map_loss = (rel_map_loss * rel_mask).sum() / rel_mask.sum()

        # Loss
        loss = rel_map_loss + ent_corres_loss if ent_corres is not None else rel_map_loss

        if return_details:
            return loss, ent_corres_pred, rel_maps_pred
        else:
            return loss

    def get_corres_tabel(self, sub_hs, obj_hs, rel_hs=None):
        """获取对应的关系表

        Args:
            sub_hs (torch.Tensor): [bsz, seq_len, h]
            obj_hs (torch.Tensor): [bsz, seq_len, h]
            rel_hs (torch.Tensor, optional): [bsz, rel_num, h]. Defaults to None.

        Returns:
            torch.Tensor: [bsz, seq_len, seq_len]
        """

        # 使用跟 PRGC 类似的处理方法，cat
        sub_hs = sub_hs.unsqueeze(2).expand(-1, -1, obj_hs.shape[1], -1)
        obj_hs = obj_hs.unsqueeze(1).expand(-1, sub_hs.shape[2], -1, -1)

        # 生成对应关系表
        corres_table = self.ent_corres(torch.cat([sub_hs, obj_hs], dim=-1))
        corres_table = corres_table.squeeze(-1)

        return corres_table

    # Train https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.training_step

    def training_step(self, batch, batch_idx):

        loss = self(batch)

        self.log('train_loss', loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, ner_maps, rel_maps, ent_corres, pos = batch
        loss, ent_corres_pred, rel_maps_pred = self(batch, return_details=True)
        self.log('val_loss', loss, on_step=True)

        pred_triples = self.deocde_triples(
            ent_corres_pred.sigmoid(), rel_maps_pred.sigmoid(), input_ids, pos)
        trgt_triples = self.deocde_triples(
            ent_corres, rel_maps, input_ids, pos)

        return {
            'pred_triplets': pred_triples,
            'trgt_triplets': trgt_triples,
        }

    def validation_epoch_end(self, outputs):

        f1, p, r = f1_score(outputs)

        self.best_f1 = max(f1, self.best_f1)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('best_f1', self.best_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask, ner_maps, rel_maps, ent_corres, pos = batch
        loss, ent_corres_pred, rel_maps_pred = self(batch, return_details=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

        pred_triples = self.deocde_triples(
            ent_corres_pred.sigmoid(), rel_maps_pred.sigmoid(), input_ids, pos)
        trgt_triples = self.deocde_triples(
            ent_corres, rel_maps, input_ids, pos)

        return {
            'pred_triplets': pred_triples,
            'trgt_triplets': trgt_triples,
        }

    def test_epoch_end(self, outputs):

        f1, p, r = f1_score(outputs)

        self.test_f1 = f1
        self.log('test_f1', f1)

    # Optimizer https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers

    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def deocde_triples(self, ent_corres, rel_maps, input_ids, pos):
        """Convert table to triplet

        Args:
            ent_corres (torch.Tensor): [bsz, seq_len, seq_len]
            rel_maps (torch.Tensor): [bsz, seq_len, rel_num * 2]
            input_ids (torch.Tensor): [bsz, seq_len]
            pos (torch.Tensor): [bsz, 4] # sent_start, sent_end, sentence_ix, sentence_start_in_doc

        """
        ent_threshold = 0.5
        rel_threshold = 0.5

        bsz, seq_len, rel_num = rel_maps.shape[0], rel_maps.shape[1], rel_maps.shape[2] // 2

        triples = set()
        for b in range(bsz):
            sent_start_token_idx, sent_end_token_idx = pos[b, 0], pos[b, 1]
            for rel in range(rel_num):
                sub_seq = rel_maps[b, :, rel]
                obj_seq = rel_maps[b, :, rel + rel_num]

                # 遍历实体序列，连续为 1 的序列为一个实体，并记录下来起始位置
                sub_entities = []  # 闭区间
                for s_i in range(sent_start_token_idx, sent_end_token_idx):
                    if sub_seq[s_i] > ent_threshold:
                        if len(sub_entities) == 0 or sub_entities[-1][1] != s_i:
                            sub_entities.append([s_i, s_i])
                        else:
                            sub_entities[-1][1] = s_i

                obj_entities = []  # 闭区间
                for o_i in range(sent_start_token_idx, sent_end_token_idx):
                    if obj_seq[o_i] > ent_threshold:
                        if len(obj_entities) == 0 or obj_entities[-1][1] != o_i:
                            obj_entities.append([o_i, o_i])
                        else:
                            obj_entities[-1][1] = o_i

                for sub in sub_entities:
                    for obj in obj_entities:
                        if ent_corres[b, sub[0], obj[0]] + ent_corres[b, sub[1], obj[1]] > rel_threshold * 2:
                            sub_token = self.tokenizer.decode(
                                input_ids[b, sub[0]:sub[1]+1])
                            obj_token = self.tokenizer.decode(
                                input_ids[b, obj[0]:obj[1]+1])
                            rel_name = self.config.dataset.rels[rel]
                            triples.add((sub_token, obj_token, rel_name))

        return triples

    # def convert_table_to_triplet(self, table, input_ids, pos):
    #     """Convert table to triplet"""
    #     bsz, seq_len = table.shape[:2]

    #     visited = torch.zeros_like(table)
    #     triples = set()

    #     for b in range(bsz):
    #         table_b = table[b]
    #         for i in range(pos[b, 0], pos[b, 1]):
    #             for j in range(pos[b, 0], pos[b, 1]):

    #                 rel = table_b[i, j]
    #                 if i == j or rel == 0 or visited[b, i, j] == 1:
    #                     continue

    #                 k = i + 1
    #                 while k < pos[b, 1] and table_b[k, j] == rel:
    #                     k += 1

    #                 l = j + 1
    #                 while l < pos[b, 1] and table_b[i, l] == rel:
    #                     l += 1

    #                 visited[b, i:k, j:l] = 1

    #                 sub = self.tokenizer.decode(input_ids[b, i:k])
    #                 obj = self.tokenizer.decode(input_ids[b, j:l])
    #                 rel = self.config.dataset.rels[rel.item()-1]
    #                 triples.add((sub, obj, rel))

    #     return triples

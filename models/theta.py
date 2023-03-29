import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.re_model import REModel
from models.ner_model import NERModel
from models.runtime_graph import RuntimeGraph
from models.functions import getBertForMaskedLMClass

from data.utils import get_language_map_dict
from utils.metrics import f1_score
from utils.optimizers import get_optimizer


class Theta(pl.LightningModule):
    """https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule"""

    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.tokenizer = data.tokenizer

        ModelClass = getBertForMaskedLMClass(config.model)
        self.plm_model = ModelClass.from_pretrained(config.model.model_name_or_path) # type: ignore

        if config.use_two_plm:
            self.plm_model_for_re = ModelClass.from_pretrained(config.model.model_name_or_path) # type: ignore

        # 常用参数
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.hidden_size = config.model.hidden_size

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Others
        self.rel_num = len(config.dataset.rels)
        self.ent_num = len(config.dataset.ents)
        self.na_idx = data.rel2id.get(config.dataset.na_label, None)

        # 模型评估
        self.best_f1 = 0
        self.test_f1 = 0
        # self.eval_fn = partial(f1_score, rel_num=config.dataset.rel_num, na_idx=self.na_idx)
        self.extand_and_init_additional_tokens()
        self.register_components()

    def extand_and_init_additional_tokens(self):
        """扩展并初始化额外的 token"""
        config = self.config

        if self.na_idx is None:
            rels = ['NA'] + config.dataset.rels
        else:
            rels = config.dataset.rels

        ents = config.dataset.ents

        rel_tokens = [f"[R{i}]" for i in range(len(rels))]
        tag_tokens = ["[SS]", "[SE]", "[OS]", "[OE]", "[ES]", "[EE]"]
        ent_tokens = ["[O]"] + [f"[B-{e}]" for e in ents] + [f"[I-{e}]" for e in ents]
        special_tokens = ["[RC]"]

        # 扩展词表
        special_tokens_dict = {'additional_special_tokens': rel_tokens + ent_tokens + tag_tokens + special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.plm_model.resize_token_embeddings(len(self.tokenizer)) # type: ignore
        if config.use_two_plm:
            self.plm_model_for_re.resize_token_embeddings(len(self.tokenizer)) # type: ignore

        self.rel_ids = self.tokenizer.convert_tokens_to_ids(rel_tokens)
        self.ent_ids = self.tokenizer.convert_tokens_to_ids(ent_tokens)
        self.tag_ids = self.tokenizer.convert_tokens_to_ids(tag_tokens)

        # 初始化关系的词向量
        with torch.no_grad():
            embeds = self.plm_model.get_input_embeddings().weight # type: ignore
            if self.config.use_two_plm:
                embeds_re = self.plm_model_for_re.get_input_embeddings().weight # type: ignore

            ace_rel_map, ace_ent_map, tag_map = get_language_map_dict()

            # 下面的代码是有点丑陋，甚至于恶心的，但是也没有办法，先这么写着吧
            # Rel
            ace_rel_ids = [self.tokenizer.encode(ace_rel_map[rel], add_special_tokens=False) for rel in rels]
            for i, rel_id in enumerate(self.rel_ids):
                embeds[rel_id] = embeds[ace_rel_ids[i]].mean(dim=-2) # type: ignore
                if self.config.use_two_plm:
                    embeds_re[rel_id] = embeds_re[ace_rel_ids[i]].mean(dim=-2) # type: ignore

            # Entity
            ace_ent_ids = [self.tokenizer.encode("outside", add_special_tokens=False)]
            for ent in ents:
                ace_ent_ids.append(self.tokenizer.encode("begin " + ace_ent_map[ent], add_special_tokens=False))

            for ent in ents:
                ace_ent_ids.append(self.tokenizer.encode("inside " + ace_ent_map[ent], add_special_tokens=False))

            for i, ent_id in enumerate(self.ent_ids):
                embeds[ent_id] = embeds[ace_ent_ids[i]].mean(dim=-2) # type: ignore
                if self.config.use_two_plm:
                    embeds_re[ent_id] = embeds_re[ace_ent_ids[i]].mean(dim=-2) # type: ignore

            # Tag
            ace_tag_ids = [self.tokenizer.encode(tag_map[tag], add_special_tokens=False) for tag in tag_tokens]
            for i, tag_id in enumerate(self.tag_ids):
                embeds[tag_id] = embeds[ace_tag_ids[i]].mean(dim=-2) # type: ignore
                if self.config.use_two_plm:
                    embeds_re[tag_id] = embeds_re[ace_tag_ids[i]].mean(dim=-2) # type: ignore

            # Special
            # TODO Task ID
            # task_id = self.tokenizer.encode("[RC]", add_special_tokens=False)[0]

            assert (self.plm_model.get_input_embeddings().weight == embeds).all() # type: ignore
            assert not self.config.use_two_plm or (
                self.config.use_two_plm and (self.plm_model_for_re.get_input_embeddings().weight == word_embeddings).all()) # type: ignore

    def register_components(self):
        """ 用于构建除了预训练语言模型之外的所有模型组件
        很多时候，使用哪些模型组件，使用哪些模块都是需要根据 config 来决定的
        """
        config = self.config

        self.rel_model = REModel(self)
        self.ner_model = NERModel(self)

        self.graph = RuntimeGraph(self) if config.use_graph else None


    def forward(self, batch, mode="train"):

        # Forward
        input_ids, attention_mask, pos, triples, ent_maps, _ = batch
        outputs = self.plm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True) # type: ignore

        # 一些参数
        hidden_state = outputs.hidden_states[-1]

        output = {}
        output["hidden_state"] = hidden_state

        # [REQUIRED] 命名实体识别损失
        ner_logits, ner_loss = self.ner_model(hidden_state, pos=pos, labels=ent_maps, graph=self.graph)
        output["ner_logits"] = ner_logits

        # [REQUIRED] 关系分类损失

        # 如果是训练阶段，使用 gold triples 计算损失，如果不是，仅保存预测的 triples 用于评估
        entities = self.ner_model.decode_entities(ent_maps, pos=pos) # gold entities
        if sum([len(e) for e in entities]) > 0:
            rel_output = self.rel_model(
                                theta=self,
                                batch=batch,
                                hidden_state=hidden_state,
                                entities=entities,
                                return_loss=True,
                                mode=mode)

            triples_pred, rel_loss, filter_loss = rel_output
            output["triples_pred_with_gold"] = triples_pred

        else:
            rel_loss = torch.tensor(0.0, device=input_ids.device)
            filter_loss = torch.tensor(0.0, device=input_ids.device)
            output["triples_pred_with_gold"] = []

        # 如果是测试阶段，使用预测的 triples
        if mode != "train":
            entities = self.ner_model.decode_entities(ner_logits, pos=pos) # pred entities
            if sum([len(e) for e in entities]) > 0:
                rel_output = self.rel_model(
                                    theta=self,
                                    batch=batch,
                                    hidden_state=hidden_state,
                                    entities=entities,
                                    return_loss=False,
                                    mode=mode)
                output["triples_pred"] = rel_output[0]
            else:
                output["triples_pred"] = []

        # 计算损失
        if mode == "train":
            loss = ner_loss + rel_loss
            self.log("ner_loss", ner_loss)
            self.log("rel_loss", rel_loss)
            self.log("filter_loss", filter_loss) # type: ignore
            loss += filter_loss

            output["loss"] = loss

        return output

    # Train https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.training_step
    def training_step(self, batch, batch_idx):

        loss = self(batch, mode="train")["loss"]
        self.log('train_loss', loss)

        lr_step = {}
        for i, pg in enumerate(self.optimizers().param_groups): # type: ignore
            lr_step[f"info/lr_{i}"] = pg["lr"]
        self.log_dict(lr_step)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, _, pos, triples, ent_maps, _  = batch
        output = self(batch, mode="dev")

        return self.eval_step_output(input_ids, pos, triples, ent_maps, output)

    def validation_epoch_end(self, outputs):
        f1, p, r = f1_score(outputs, 'pred_triples', 'gold_triples')
        ner_f1, ner_p, ner_r = f1_score(outputs, 'pred_entities', 'gold_entities')
        rel_f1, rel_p, rel_r = f1_score(outputs, 'pred_triples_with_gold', 'gold_triples')

        self.best_f1 = max(f1, self.best_f1)
        self.log_dict_values({'val_p': p, 'val_r': r})
        self.log_dict_values({'val_f1': f1, 'best_f1': self.best_f1}, on_epoch=True, prog_bar=True)
        self.log_dict_values({'val_ner_f1': ner_f1, 'val_ner_p': ner_p, 'val_ner_r': ner_r})
        self.log_dict_values({'val_rel_f1': rel_f1, 'val_rel_p': rel_p, 'val_rel_r': rel_r})

    def test_step(self, batch, batch_idx):
        input_ids, _, pos, triples, ent_maps, _  = batch
        output = self(batch, mode="test")

        return self.eval_step_output(input_ids, pos, triples, ent_maps, output)

    def test_epoch_end(self, outputs):
        f1, p, r = f1_score(outputs, 'pred_triples', 'gold_triples')
        ner_f1, ner_p, ner_r = f1_score(outputs, 'pred_entities', 'gold_entities')
        rel_f1, rel_p, rel_r = f1_score(outputs, 'pred_triples_with_gold', 'gold_triples')

        self.test_f1 = f1
        self.log_dict_values({'test_f1': f1, 'test_p': p, 'test_r': r})
        self.log_dict_values({'test_ner_f1': ner_f1, 'test_ner_p': ner_p, 'test_ner_r': ner_r})
        self.log_dict_values({'test_rel_f1': rel_f1, 'test_rel_p': rel_p, 'test_rel_r': rel_r})

    def eval_step_output(self, input_ids, pos, triples, ent_maps, output):
        pred_entities = self.ner_model.decode_entities(output["ner_logits"], pos=pos)
        gold_entities = self.ner_model.decode_entities(ent_maps, pos=pos)
        pred_entities = self.get_span_set(input_ids, pred_entities)
        gold_entities = self.get_span_set(input_ids, gold_entities)

        pred_triples, gold_triples = self.get_triple_set(input_ids, triples, output, "triples_pred")
        pred_triples_with_gold = self.get_triple_set(input_ids, triples, output, "triples_pred_with_gold", pred_only=True)

        return {
            'pred_entities': pred_entities,
            'gold_entities': gold_entities,
            'pred_triples': pred_triples,
            'gold_triples': gold_triples,
            'pred_triples_with_gold': pred_triples_with_gold,
        }


    # Optimizer https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def get_triple_set(self, input_ids, triples, output, name, pred_only=False):
        pred_triples = set()
        # start, end 是左闭右开区间
        # [batch_idx, sub_start, sub_end, obj_start, obj_end, sub_type, obj_type, rel_idx]
        #  0          1          2        3          4        5         6         7 (include NA)
        for t in output[name]:
            if t[7] != 0:
                sub_token = self.tokenizer.decode(input_ids[t[0], t[1]:t[2]])
                obj_token = self.tokenizer.decode(input_ids[t[0], t[3]:t[4]])
                rel_type = self.config.dataset.rels[t[7]-1]
                sub_type = self.config.dataset.ents[t[5]]
                obj_type = self.config.dataset.ents[t[6]]
                triple = (sub_token, obj_token, rel_type, sub_type, obj_type)
                if self.config.use_rel_strict:
                    pred_triples.add(triple)
                else:
                    pred_triples.add(triple[:3])

        if pred_only:
            return pred_triples

        gold_triples = set()
        # [sub_start, sub_end, obj_start, obj_end, rel_idx, sub_type, obj_type]
        #  0          1        2          3        4(No NA) 5         6
        for b in range(input_ids.shape[0]):
            for t in triples[b]:
                if t[4] != -1:
                    sub_token = self.tokenizer.decode(input_ids[b, t[0]:t[1]])
                    obj_token = self.tokenizer.decode(input_ids[b, t[2]:t[3]])
                    rel_type = self.config.dataset.rels[t[4]]
                    sub_type = self.config.dataset.ents[t[5]]
                    obj_type = self.config.dataset.ents[t[6]]
                    triple = (sub_token, obj_token, rel_type, sub_type, obj_type)
                    if self.config.use_rel_strict:
                        gold_triples.add(triple)
                    else:
                        gold_triples.add(triple[:3])
                else:
                    break
        return pred_triples,gold_triples

    def get_span_set(self, input_ids, entities):
        entities_token = set()
        for b in range(input_ids.shape[0]):
            for e in entities[b]:
                ent_token = self.tokenizer.decode(input_ids[b, e[0]:e[1]])
                ent_type = self.config.dataset.ents[e[2]]
                entities_token.add((ent_token, ent_type))
        return entities_token

    def log_dict_values(self, d, **kwargs):
        for k, v in d.items():
            self.log(k, v, **kwargs)


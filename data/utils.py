import numpy as np

import torch

def convert_dataset_to_samples(dataset, config, tokenizer, is_test=False):
    """
    Extract sentences and gold entities from a dataset

    关于 context_window 和 max_seq_len 的设计考虑：
    1. context_window 用于指定上下文窗口的大小
    2. max_seq_len 用于指定最大序列长度
    对于每一个句子，首先将其转化为 tokens，然后考虑分别向前和向后扩展
    这里的扩展是指将句子的 tokens 从中间向两边扩展，直到达到 context_window 的大小
    由于句子并不一定能够向前或者向后扩展到 context_window 的大小，比如文档开头和结尾的句子
    因此，实际上 context_window 的大小可能会小于指定的大小，然后填充到 max_seq_len 的大小
    同时需要注意的是，如果 context_window 大于 max_seq_len，那么 context_window 的大小会被忽略
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)

    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0

    ner2id = {name: idx for idx, name in enumerate(config.dataset.ents)}
    rel2id = {name: idx for idx, name in enumerate(config.dataset.rels)}

    max_seq_len = config.get("max_seq_len", config.dataset.max_seq_len) # 最长的句子序列是 103
    context_window = config.get("context_window", 0)
    # max_span_length = config.get("max_span_length", 10)

    if context_window and context_window > max_seq_len:
        print(f'context_window{context_window} > max_seq_len{max_seq_len}, context window will be set as `max_seq_len - 2`')
        context_window = max_seq_len - 2

    split = config.dataset.get("split", 0)

    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset)*0.9))
    elif split == 2:
        data_range = (int(len(dataset)*0.9), len(dataset))

    # c means the index
    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            num_ner += len(sent.ner)

            sample = {}
            sample['tokens'] = sent.text

            sent_length = len(sent.text)

            if context_window and sent_length > context_window:
                print('Long sentence: {} {}'.format(sample, sent_length))

            max_len = max(max_len, sent_length)
            max_ner = max(max_ner, len(sent.ner))

            # tokenized tokens
            start2idx = []
            end2idx = []

            tokenized_tokens = []
            for token in sample['tokens']:
                tokenized_token = tokenizer.tokenize(token)
                tokenized_tokens += tokenized_token
                start2idx.append(len(tokenized_tokens)-len(tokenized_token))
                end2idx.append(len(tokenized_tokens)) # 左闭右开

            sent_start = 0
            sent_end = len(tokenized_tokens)

            # 根据窗口进行扩充
            if context_window:

                # 填充 left
                add_left = (context_window-len(tokenized_tokens)) // 2

                left_tokens = []
                left_sent_idx = sent.sentence_ix - 1
                while left_sent_idx >= 0 and add_left > 0:
                    left_token_to_add = tokenizer.tokenize(" ".join(doc[left_sent_idx].text))[-add_left:]
                    left_tokens = left_token_to_add + left_tokens
                    add_left -= len(left_token_to_add)
                    left_sent_idx -= 1

                sent_start = len(left_tokens)
                tokenized_tokens = left_tokens + tokenized_tokens

                # 填充 right
                add_right = context_window - len(tokenized_tokens)

                right_tokens = []
                right_sent_idx = sent.sentence_ix + 1
                while right_sent_idx < len(doc) and add_right > 0:
                    right_token_to_add = tokenizer.tokenize(" ".join(doc[right_sent_idx].text))[:add_right]
                    right_tokens = right_tokens + right_token_to_add
                    add_right -= len(right_token_to_add)
                    right_sent_idx += 1

                tokenized_tokens = tokenized_tokens + right_tokens

            # 补全到 max_seq_len
            if len(tokenized_tokens) < max_seq_len - 2:

                attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
                attention_mask[:len(tokenized_tokens)+2] = 1.0

                tokenized_tokens = [tokenizer.cls_token] + tokenized_tokens + [tokenizer.sep_token]
                tokenized_tokens += [tokenizer.pad_token] * (max_seq_len - len(tokenized_tokens))

            else:

                attention_mask = torch.ones(max_seq_len, dtype=torch.long)
                tokenized_tokens = [tokenizer.cls_token] + tokenized_tokens[:max_seq_len-2] + [tokenizer.sep_token]

            assert len(tokenized_tokens) == len(attention_mask) == max_seq_len

            input_ids = tokenizer.convert_tokens_to_ids(tokenized_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            assert sum(input_ids != tokenizer.pad_token_id) == sum(attention_mask != 0)

            # 原本的 token 在现有的 tokens 序列中的起始位置，左闭右开
            sent_start += 1         # sent_start 不包含 cls token 或者 context window 的 token
            sent_end += sent_start  # sent_end 包含 sep token 或者 context window 的 token

            # 实体的位置索引映射
            start2idx = [i + sent_start for i in start2idx]
            end2idx = [i + sent_start for i in end2idx] # 因为有 cls token

            """关于命名实体识别的说明
            Rules:
                - 使用 BIO 的标注方式
                - 如果填充的是 0 则表示不是实体

            Shape: [max_seq_len]
            """
            ent_type = {}
            ner_total_len = 0  # 记录 ner 的总长度，用于判断是否有重叠
            added = []  # 记录已经添加过的 ner，因为在文档 323 中存在同一个实体标注了两次，两次是不同的类型，这里以第二个类型为准
            ent_maps = torch.zeros(len(tokenized_tokens), dtype=torch.long) # 0 表示不是实体
            for ner in sent.ner:
                ent_s = start2idx[ner.span.start_sent]
                ent_e = end2idx[ner.span.end_sent]
                # 实体边界识别 O B-0 B-1 B-2 B-3 B-4 B-5 B-6 I-0 I-1 I-2 I-3 I-4 I-5 I-6
                #            0 1   2   3   4   5   6   7   8   9   10  11  12  13  14
                if config.use_span:
                    ent_maps[ent_s] = 1 # B
                    ent_maps[ent_s+1:ent_e] = 2 # I
                elif config.use_ner:
                    ent_maps[ent_s] = ner2id[ner.label] + 1
                    ent_maps[ent_s+1:ent_e] = ner2id[ner.label] + len(ner2id) + 1 # len(ner2id) + 1 表示 I, len(ner2id) == 7

                if ner.span not in added:
                    ner_total_len += ent_e - ent_s
                    added.append(ner.span)

                ent_type[(ent_s, ent_e)] = ner2id[ner.label]

            assert ner_total_len == (ent_maps != 0).sum(), 'NER overlap'

            ent_corres = torch.zeros((max_seq_len, max_seq_len), dtype=torch.long)
            triples = set()

            for rel in sent.relations:

                sub_s = start2idx[rel.pair[0].start_sent]
                sub_e = end2idx[rel.pair[0].end_sent] # 开区间
                obj_s = start2idx[rel.pair[1].start_sent]
                obj_e = end2idx[rel.pair[1].end_sent] # 开区间
                sub_type = ent_type[(sub_s, sub_e)]
                obj_type = ent_type[(obj_s, obj_e)]
                triples.add((sub_s, sub_e, obj_s, obj_e, rel2id[rel.label], sub_type, obj_type))

                ent_corres[sub_s:sub_e, obj_s:obj_e] = 1

            max_tripes_count = config.get("max_tripes_count", 30)
            assert len(triples) <= max_tripes_count, f"triples count {len(triples)} > {max_tripes_count}"
            # 补全到最大的三元组数量
            triples = list(triples) + [(-1, -1, -1, -1, -1, -1, -1)] * (max_tripes_count - len(triples))

            sample['input_ids'] = input_ids
            sample['triples'] = torch.tensor(triples, dtype=torch.long)
            sample['attention_mask'] = attention_mask
            sample['pos'] = torch.tensor((sent_start, sent_end, sent.sentence_ix, sent.sentence_start), dtype=torch.long)
            sample['ent_maps'] = ent_maps
            sample['ent_corres'] = ent_corres
            samples.append(sample)

    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    print('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length'%(len(samples), data_range[1]-data_range[0], num_ner, avg_length, max_length))
    print('Max Length: %d, max NER: %d'%(max_len, max_ner))

    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
    triples = torch.stack([sample["triples"] for sample in samples])
    pos = torch.stack([sample["pos"] for sample in samples])

    if config.use_span or config.use_ner:
        ent_maps = torch.stack([sample["ent_maps"] for sample in samples])
    else:
        ent_maps = torch.zeros((len(samples)), dtype=torch.long)

    # if config.use_ent_corres:
    #     ent_corres = torch.stack([sample["ent_corres"] for sample in samples])
    # else:
    #     ent_corres = torch.zeros((len(samples)), dtype=torch.long)

    ent_corres = torch.zeros((len(samples)), dtype=torch.long)


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pos": pos,
        "triples": triples,
        "ent_maps": ent_maps,
        "ent_corres": ent_corres,
    }


def get_language_map_dict():
    ace_rel_map = {
        'NA': 'no relation in this sentence',
        'ART': 'artifact person',
        'ORG-AFF': 'organization affiliation',
        'GEN-AFF': 'general affiliation',
        'PHYS': 'physical location',
        'PER-SOC': 'personal social',
        'PART-WHOLE': 'part whole'
    }

    # ['NA', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
    ace_ent_map = {
        'NA': 'none',
        'FAC': 'entity facility',
        'WEA': 'entity weapon',
        'LOC': 'entity location',
        'VEH': 'entity vehicle',
        'GPE': 'entity geopolitical',
        'ORG': 'entity organization',
        'PER': 'entity person'
    }

    tag_map = {
        '[SS]': 'subject start',
        '[SE]': 'subject end',
        '[OS]': 'object start',
        '[OE]': 'object end',
        '[ES]': 'entity start',
        '[EE]': 'entity end'
    }

    return ace_rel_map,ace_ent_map,tag_map



# def extend_maps_to_one_hot(ent_maps, rel_maps, config):
#     """为了方便跟模型计算的结果比较，转为为 one-hot 的形式"""
#     # trans ner maps from b * max_seq_len to b * max_seq_len * len(config.dataset.ents)
#     ent_maps += 1  # 防止 np.log2(0) 报错
#     bsz, seq_len = ent_maps.size()

#     new_span_maps = []
#     for b in range(bsz):
#         span_maps_b = np.zeros([seq_len, len(config.dataset.ents)], dtype=int)
#         for i in range(len(config.dataset.ents)):
#             span_maps_b[:,i] = np.int_((np.log2(ent_maps[b].cpu())-1)==i)
#         new_span_maps.append(span_maps_b)
#     ent_maps = torch.tensor(new_span_maps)

#     # trans rel maps from b * max_seq_len * max_seq_len to b * max_seq_len * max_seq_len * len(config.dataset.rels)
#     rel_maps += 1  # 防止 np.log2(0) 报错
#     bsz, seq_len, _ = rel_maps.size()

#     new_rel_maps = []
#     for b in range(bsz):
#         rel_maps_b = np.zeros([seq_len, seq_len, len(config.dataset.rels)], dtype=int)
#         for i in range(len(config.dataset.rels)):
#             rel_maps_b[:,:,i] = np.int_((np.log2(rel_maps[b].cpu())-1)==i)
#         new_rel_maps.append(rel_maps_b)
#     rel_maps = torch.tensor(new_rel_maps)

#     return ent_maps, rel_maps


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)

# def get_train_fold(data, fold):
#     print('Getting train fold %d...'%fold)
#     l = int(len(data) * 0.1 * fold)
#     r = int(len(data) * 0.1 * (fold+1))
#     new_js = []
#     new_docs = []
#     for i in range(len(data)):
#         if i < l or i >= r:
#             new_js.append(data.js[i])
#             new_docs.append(data.documents[i])
#     print('# documents: %d --> %d'%(len(data), len(new_docs)))
#     data.js = new_js
#     data.documents = new_docs
#     return data

# def get_test_fold(data, fold):
#     print('Getting test fold %d...'%fold)
#     l = int(len(data) * 0.1 * fold)
#     r = int(len(data) * 0.1 * (fold+1))
#     new_js = []
#     new_docs = []
#     for i in range(len(data)):
#         if i >= l and i < r:
#             new_js.append(data.js[i])
#             new_docs.append(data.documents[i])
#     print('# documents: %d --> %d'%(len(data), len(new_docs)))
#     data.js = new_js
#     data.documents = new_docs
#     return data


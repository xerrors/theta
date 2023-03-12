def f1_score(outputs, pred_name, gold_name):
    """计算 F1 分数"""
    pred = 0
    gold = 0
    correct = 0

    for val_out in outputs:
        pred_triples = set(val_out[pred_name])
        gold_triples = set(val_out[gold_name])
        pred += len(pred_triples)
        gold += len(gold_triples)
        correct += len(set(pred_triples) & set(gold_triples))

    precision = correct / (pred + 1e-8)
    recall = correct / (gold + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


# def f1_score(outputs, consider_conf=True, threshold=0.5):
#     """计算 F1 分数
#     Args:
#         outputs (list): 预测结果
#         consider_conf (bool, optional): 是否使用置信度. Defaults to False.
#         threshold (float, optional): 置信度阈值. Defaults to 0.5.
#     """
#     pred = 0
#     gold = 0
#     correct = 0

#     for val_out in outputs:
#         if list(val_out['gold_triples']) == [] and list(val_out['pred_triples']) == []:
#             continue

#         triple_length = len(list(val_out['pred_triples'] | val_out['gold_triples'])[0])

#         # 第一种情况，输出里面并不存在置信度，直接使用预测结果
#         if triple_length == 3:
#             pred_triples = set(val_out['pred_triples'])
#             gold_triples = set(val_out['gold_triples'])
#         # 第二种情况，输出里面存在置信度，需要根据置信度来判断是否是预测结果
#         elif triple_length == 4:
#             if consider_conf:
#                 pred_triples = set([x[:3] for x in val_out['pred_triples'] if x[3] > threshold])
#             else:
#                 pred_triples = set([x[:3] for x in val_out['pred_triples']])
#             gold_triples = set([x[:3] for x in val_out['gold_triples']])
#         else:
#             raise ValueError('Invalid triples.')

#         pred += len(pred_triples)
#         gold += len(gold_triples)
#         correct += len(set(pred_triples) & set(gold_triples))

#     precision = correct / (pred + 1e-8)
#     recall = correct / (gold + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)
#     return f1, precision, recall

# def f1_score(output, label, rel_num=42, na_num=None):
#     """计算 F1 分数
#     Args:
#         output (ndarray): 预测结果
#         label (ndarray): 真实标签
#         rel_num (int, optional): 关系数量. Defaults to 42.
#         na_num (int, optional): NA 的 id. Defaults to None.
#     """
#     gold_by_relation = [0] * rel_num
#     guess_by_relation = [0] * rel_num
#     correct_by_relation = [0] * rel_num

#     if output.shape != label.shape:
#         output = np.argmax(output, axis=-1)

#     for i in range(len(output)):
#         guess = output[i]
#         gold = label[i]

#         # re index,
#         if na_num is not None:
#             if guess == na_num:
#                 guess = 0
#             elif guess < na_num:
#                 guess += 1

#             if gold == na_num:
#                 gold = 0
#             elif gold < na_num:
#                 gold += 1

#         # 把 NA 的 id 当作是 0， guess 和 gold 可以被看成是 0
#         guess_by_relation[guess] += 1
#         gold_by_relation[gold] += 1
#         if gold == guess:
#             correct_by_relation[gold] += 1

#     f1_by_relation = [0] * rel_num
#     recall_by_relation = [0] * rel_num
#     prec_by_relation = [0] * rel_num

#     for i in range(rel_num):

#         recall = 0
#         if gold_by_relation[i] > 0:
#             recall = correct_by_relation[i] / gold_by_relation[i]

#         precision = 0
#         if guess_by_relation[i] > 0:
#             precision = correct_by_relation[i] / guess_by_relation[i]

#         if recall + precision > 0 :
#             f1_by_relation[i] = 2 * recall * precision / (recall + precision)

#         recall_by_relation[i] = recall
#         prec_by_relation[i] = precision

#     micro_f1, recall, prec = 0, 0, 0
#     if sum(guess_by_relation[1:]) != 0 and sum(correct_by_relation[1:]) != 0:
#         recall = sum(correct_by_relation[1:]) / sum(gold_by_relation[1:])
#         prec = sum(correct_by_relation[1:]) / sum(guess_by_relation[1:])
#         micro_f1 = 2 * recall * prec / (recall+prec)

#     return dict(f1=micro_f1, r=recall, p=prec,
#                 f1_by_relation=f1_by_relation,
#                 recall_by_relation=recall_by_relation,
#                 prec_by_relation=prec_by_relation,
#                 gold_by_relation=gold_by_relation,
#                 guess_by_relation=guess_by_relation,
#                 correct_by_relation=correct_by_relation)

def f1_score(outputs):
    pred = 0
    trgt = 0
    correct = 0

    for val_out in outputs:
        pred_triplets = val_out['pred_triplets']
        trgt_triplets = val_out['trgt_triplets']

        pred += len(pred_triplets)
        trgt += len(trgt_triplets)
        correct += len(set(pred_triplets) & set(trgt_triplets))

    precision = correct / (pred + 1e-8)
    recall = correct / (trgt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall

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

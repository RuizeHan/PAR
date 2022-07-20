import numpy as np
from cdp.source import graph


def evaluate_group_detection(preds: list, gts: list, step=0.01, max_person=5):
    tp = 0
    fp = 0
    fn = 0

    def generate_list(a: np.ndarray):
        res = []
        searched = []
        N, _ = a.shape
        for i in range(N):
            if i not in searched:
                temp = [i]
                for j in range(N):
                    if a[i, j] != 0 and i != j:
                        temp.append(j)
                if len(temp) > 1:
                    res.append(temp)
                for k in temp:
                    searched.append(k)
        return res

    def group_eval(GROUP, GT):
        TP = 0

        for gt in GT:
            gt_set = set(gt)
            gt_card = len(gt)
            for group in GROUP:
                group_set = set(group)
                group_card = len(group)
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 1 / 2:
                    TP += 1

        FP = len(GROUP) - TP
        FN = len(GT) - TP

        return TP, FP, FN

    gt = [generate_list(i) for i in gts]
    pred = [generate_list(i) for i in preds]

    for i in range(len(pred)):  # vfid is video id.
        TP, FP, FN = group_eval(pred[i], gt[i])
        tp += TP
        fn += FN
        fp += FP

    safed = lambda x, y: 0 if y == 0 else x / y
    prec = safed(tp, (tp + fp))
    recall = safed(tp, (tp + fn))
    f1 = safed(2 * prec * recall, (prec + recall))
    return prec, recall, f1


def evaluate_social_activity(acty_gt, acty_predict, group_gt, group_predict):
    assert len(acty_gt) == len(acty_predict) == len(group_predict) == len(group_gt)

    Precision = 0
    Recall = 0
    F1 = 0
    Group_num = 0

    num_frames = len(acty_gt)
    for i in range(num_frames):
        this_precision, this_recall, this_f1, this_group_num = social_activity_eval(group_predict[i],
                                                                                    group_gt[i],
                                                                                    acty_predict[i],
                                                                                    acty_gt[i])
        Precision += this_precision
        Recall += this_recall
        F1 += this_f1
        Group_num += this_group_num

    Precision = Precision / Group_num
    Recall = Recall / Group_num
    F1 = F1 / Group_num

    return Precision, Recall, F1


def social_activity_eval(GROUP, GT, acty_predict, acty_gt):
    precision = 0
    recall = 0
    f1 = 0
    group_num = len(GROUP)

    for gt_index in range(len(GT)):
        gt_set = set(GT[gt_index])
        gt_card = len(GT[gt_index])
        for predict_index in range(len(GROUP)):
            group_set = set(GROUP[predict_index])
            group_card = len(GROUP[predict_index])
            inters = list(gt_set & group_set)
            inters_card = len(inters)
            if group_card == 2 and gt_card == 2:
                if not len(gt_set - group_set):
                    if sum(acty_gt[gt_index]) != 0:
                        recall += sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index])) / sum(
                            acty_gt[gt_index])
                    if sum(acty_predict[predict_index]) != 0:
                        precision += sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index])) / sum(
                            acty_predict[predict_index])
                    p = sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index]))
                    q = sum(acty_predict[predict_index]) + sum(acty_gt[gt_index])
                    f1 += (2 * p) / q
            elif inters_card / max(gt_card, group_card) > 1 / 2:
                if sum(acty_gt[gt_index]) != 0:
                    recall += sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index])) / sum(
                        acty_gt[gt_index])
                if sum(acty_predict[predict_index]) != 0:
                    precision += sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index])) / sum(
                        acty_predict[predict_index])
                p = sum(np.logical_and(acty_gt[gt_index], acty_predict[predict_index]))
                q = sum(acty_predict[predict_index]) + sum(acty_gt[gt_index])
                f1 += (2 * p) / q

    return precision, recall, f1, group_num

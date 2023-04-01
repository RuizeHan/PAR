import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.linear_assignment_ import linear_assignment
from my_utils.group_detection_PRF import *
import torch
import argparse


def evaluate_group_detection(preds: list, gts: list):
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

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    print(w)
    for i in range(y_pred.size):
        print(y_pred[i], y_true[i])
        w[y_pred[i], y_true[i]] += 1
        print(w)
        assert False

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def generate_labels(scores, threshold):
    result = np.zeros((len(scores), len(scores[0])))
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            if scores[i][j] > threshold:
                result[i][j] = 1
    return result


def concat(individual_pre, num):
    individual_pre_all = []
    for i in range(len(individual_pre)):
        temp = individual_pre[i].reshape(-1, num)
        for j in range(temp.shape[0]):
            individual_pre_all.append(np.array(temp[j]))
    return individual_pre_all

if __name__=='__main__':
    parser = argparse.ArgumentParser('Please input the path of the result pt file:')
    parser.add_argument('--pt_path', type=str)
    args = parser.parse_args()
   
    data = np.load(args.pt_path, allow_pickle=True).item()

    # ====================================================================================================================
    individual_pre = data['action_result'][0]
    individual_gt = data['action_result'][1]
    precision, recall, f1, _ = precision_recall_fscore_support(individual_gt, individual_pre, sample_weight=None,
                                                               average='samples')
    print('individual action:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    ga_pre = data['social_activity_result']['social_acty_predict']
    ga_gt = data['social_activity_result']['social_acty_gt']
    gd_gt = data['social_activity_result']['group_det_gt']
    gd_pre = data['social_activity_result']['group_det_predict']
    precision, recall, f1 = evaluate_social_activity(ga_gt, ga_pre, gd_gt, gd_pre)

    print('social group activity:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    oa_pre = data['activity_result'][1]
    oa_gt = data['activity_result'][0]
    precision, recall, f1, _ = precision_recall_fscore_support(oa_gt, oa_pre, sample_weight=None,
                                                               average='samples')

    print('global activity:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    gd_pre = (data['group_detection_result']['prediction'])
    gd_gt = (data['group_detection_result']['GT'])

    _, _, acc = evaluate_group_detection(gd_pre, gd_gt)

    print('group detection acc:', acc)

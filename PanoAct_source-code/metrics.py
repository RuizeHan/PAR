import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.linear_assignment_ import linear_assignment
import torch
import argparse


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
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

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
    parser.add_argument('pt_path', type=str)

    args = parser.parse_args()
    data = torch.load(args.pt_path)

    # ====================================================================================================================
    individual_pre = data['individual']['prediction']
    individual_gt = data['individual']['gt']
    individual_pre = concat((individual_pre), 27)
    individual_gt = concat((individual_gt), 27)
    individual_pre = generate_labels(individual_pre, 0.5)
    precision, recall, f1, _ = precision_recall_fscore_support(individual_gt, individual_pre, sample_weight=None,
                                                               average='samples')
    print('individual action:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    ga_pre = data['ga']['prediction']
    ga_gt = data['ga']['gt']
    ga_pre = concat((ga_pre), 32)
    ga_gt = concat((ga_gt), 32)
    ga_pre = generate_labels(ga_pre, 0.5)
    precision, recall, f1, _ = precision_recall_fscore_support(ga_gt, ga_pre, sample_weight=None,
                                                               average='samples')

    print('social group activity:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    oa_pre = data['oa']['prediction']
    oa_gt = data['oa']['gt']
    oa_pre = concat((oa_pre), 7)
    oa_gt = concat((oa_gt), 7)
    oa_pre = generate_labels(oa_pre, 0.5)
    precision, recall, f1, _ = precision_recall_fscore_support(oa_gt, oa_pre, sample_weight=None,
                                                               average='samples')

    print('global activity:')
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # ====================================================================================================================
    gd_pre = (data['gd']['prediction'])
    gd_gt = (data['gd']['gt'])
    for i in range(len(gd_pre)):
        gd_pre[i] = np.array(gd_pre[i])
        gd_gt[i] = np.array(gd_gt[i])

    result = []
    num = []
    for i in range(len(gd_pre)):
        gt = gd_gt[i]
        pre = gd_pre[i]
        group_detection_acc = acc(gt, pre)
        result.append(group_detection_acc)
        num.append(len(gt))

    all = 0
    counter = 0
    for i in range(len(result)):
        counter += result[i] * num[i]
        all += num[i]

    print('group detection acc:', counter / all)

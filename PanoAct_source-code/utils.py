import torch
import time
import itertools
import numpy as np
from torch.nn.functional import cosine_similarity


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k, ': ', v)
    print_log(cfg.log_path, '======================End=======================')


def show_epoch_info(phase, log_path, info, cfg):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print('experiment_index: ' + str(cfg.experiment_index))
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))

    if phase == 'Test':
        print_log(log_path,
                  'Global f1: %.2f%%,action f1: %.2f%%,social activity f1: %.2f%%, group_det f1: %.2f%%, Overall f1: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
                      info['activities_f1'], info['actions_f1'], info['social_activities_f1'],
                      info['group_detection_acc'], info['overall_f1'],info['loss'],
                      info['time']))
        print_log(log_path,
                  'Group_Activity p: %.2f%%,action p: %.2f%%,social activity p: %.2f%%' % (
                      info['activities_p'], info['actions_p'], info['social_activities_p']))
        print_log(log_path,
                  'Group_Activity r: %.2f%%,action r: %.2f%%,social activity r: %.2f%%' % (
                      info['activities_r'], info['actions_r'], info['social_activities_r']))

    else:
        print_log(log_path,
                  'Global f1: %.2f%%,action f1: %.2f%%, social activity f1: %.2f%%, Overall f1: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
                      info['activities_f1'], info['actions_f1'], info['social_activities_f1'], info['overall_f1'],info['loss'],
                      info['time']))


def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg = ['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                      'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                      'result_path', 'log_path', 'device']

    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)

        for k, v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print(k, ': ', v, file=f)

        print('=====================Result======================', file=f)

        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)

        print('Cost total %.4f hours.' % (exp_result['total_time']), file=f)

        print('======================End=======================', file=f)

    data_dict = pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name] = exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def create_P_matrix(pred):
    people_num = max(pred) + 1
    bbox_num = len(pred)
    matrix_temp = np.identity(bbox_num)
    for i in range(people_num):
        temp = []
        for j in range(bbox_num):
            if pred[j] == i:
                temp.append(j)
        cc = list(itertools.combinations(temp, 2))
        for (p, q) in cc:
            matrix_temp[p][q] = 1
            matrix_temp[q][p] = 1

    return matrix_temp


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def generate_labels(scores, threshold):
    result = np.zeros((scores.shape[0], scores.shape[1]))
    # result = [np.zeros(scores.shape[1],) for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if scores[i][j] > threshold:
                result[i][j] = 1
    return result


# calculate the laplacian matrix
def get_laplacian(matrix_A):
    matrix_D = torch.sum(matrix_A, dim=1)
    matrix_D = torch.diag(matrix_D)
    matrix_L = matrix_D - matrix_A
    return matrix_L


# get the eig loss in jrdb paper
def get_eig_loss(predict_matrix, gt_matrix, device):
    eig_loss = torch.zeros((1,), requires_grad=True).to(device=device)
    for i in range(len(predict_matrix)):
        this_num = int(gt_matrix[i].shape[1] ** 0.5)
        this_gt_matrix = torch.tensor(gt_matrix[i]).reshape((this_num, this_num)).to(device=device)
        this_predict_matrix = predict_matrix[i].reshape((this_num, this_num))
        this_laplacian_matrix = get_laplacian(this_predict_matrix).double()

        # get eigenvectors and eigenvalues
        (evals, evecs) = torch.eig(this_gt_matrix, eigenvectors=True)

        # get zero eigenvectors of this gt matrix
        zero_evecs = []
        for val in range(evals.shape[0]):
            # find zero eigenvalues
            if torch.abs(evals[val][0]).item() == 0 and torch.abs(evals[0][1]).item() == 0:
                zero_evecs.append(evecs[val].double())

        if len(zero_evecs) > 0:
            for this_zero_evec in zero_evecs:
                temp = torch.mm(this_zero_evec.reshape(1, -1), this_laplacian_matrix.t())
                temp = torch.mm(temp, this_laplacian_matrix)
                this_loss_1 = torch.mm(temp, this_zero_evec.reshape(1, -1).t())
                this_loss_2 = 1 / torch.exp(torch.trace(torch.mm(this_laplacian_matrix.t(), this_laplacian_matrix)))
                this_loss = this_loss_1 + this_loss_2
                eig_loss += this_loss.reshape(1, )
        return eig_loss


def cosine(source, target):
    return (1 - cosine_similarity(source, target)).mean()


# def calc_pairwise_distance_3d(X, Y):
#     """
#     computes pairwise distance between each element
#     Args: 
#         X: [B,N,D]
#         Y: [B,M,D]
#     Returns:
#         dist: [B,N,M] matrix of euclidean distances
#     """
#     B=X.shape[0]
    
#     rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
#     ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
#     dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
#     if True in torch.isnan(dist):
#         print('debug')
#     return torch.sqrt(dist)

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args:
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    # B = X.shape[0]

    # rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    # ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    # dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    X = X.reshape(-1, 2)
    Y = Y.reshape(-1, 2)
    num_n = X.shape[0]
    result = torch.zeros((1, num_n, num_n))
    for i in range(num_n):
        for j in range(num_n):
            temp = (X[i][0] - Y[j][0]) ** 2 + (X[i][1] - Y[j][1]) ** 2
            temp = torch.sqrt(temp)
            result[0][i][j] = temp

    result_original = torch.sqrt(result)
    if True in torch.isnan(result_original):
        print('debug')
    return result_original


def calc_foot_distance(boxes_positions):
    boxes_positions = boxes_positions.reshape(-1, 4)
    num_n = boxes_positions.shape[0]
    xy_foot = torch.zeros((num_n, 2))
    xy_foot[:, 0] = (boxes_positions[:, 0] + boxes_positions[:, 2]) / 2
    xy_foot[:, 1] = boxes_positions[:, 3]

    result = torch.zeros((1, num_n, num_n))
    for i in range(num_n):
        for j in range(num_n):
            temp = min(abs(xy_foot[i][0] - xy_foot[j][0]), (467 - abs(xy_foot[i][0] - xy_foot[j][0]))) ** 2 + \
                   (xy_foot[i][1] - xy_foot[j][1]) ** 2
            temp = torch.sqrt(temp)
            result[0][i][j] = temp

    return result


def normalize_distance(boxes_positions, boxes_distance_foot):
    boxes_positions = boxes_positions.reshape(-1, 4)
    num_n = boxes_positions.shape[0]
    for i in range(num_n):
        for j in range(num_n):
            sum_area_i = (abs(boxes_positions[i][2] - boxes_positions[i][0]) * abs(
                boxes_positions[i][3] - boxes_positions[i][1]))
            sum_area_j = (abs(boxes_positions[j][2] - boxes_positions[j][0]) * abs(
                boxes_positions[j][3] - boxes_positions[j][1]))
            boxes_distance_foot[0][i][j] = boxes_distance_foot[0][i][j] / torch.sqrt(sum_area_i + sum_area_j)
    return boxes_distance_foot


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


def generate_social_acty_labels(social_activity_in, group_id_in, is_train):
    social_group_gt = create_P_matrix(np.array(group_id_in.cpu()))
    social_group_gt = generate_list(social_group_gt)
    social_group_gt.sort()

    social_activity_in_new = []
    for group in social_group_gt:
        if is_train:
            social_activity_in_new.append(social_activity_in[group[0]])
        else:
            social_activity_in_new.append(np.array(social_activity_in[group[0]].cpu()))

    return social_activity_in_new, social_group_gt

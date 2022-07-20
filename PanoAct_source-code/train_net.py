import torch.optim as optim
from dataset import *
from models.block_model import *
from base_model import *
from utils import *
import numpy as np
from spectral_cluster.spectralcluster import SpectralClusterer
from my_utils import clustering_acc
import warnings
from sklearn.metrics import precision_recall_fscore_support
from my_utils.group_detection_PRF import evaluate_group_detection, evaluate_social_activity
from my_utils.multi_label_focal_loss import FocalLoss_1, FocalLoss_MultiLabel
from my_utils.equalized_focal_loss import EqualizedFocalLoss
from spectral_cluster.spectralcluster.refinement import *
from spectral_cluster.spectralcluster import autotune
from my_utils.resample_loss import ResampleLoss
from my_utils.long_tail_losses import *


def train_net(cfg):
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    training_set, validation_set = return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 2
    }
    training_loader = data.DataLoader(training_set, **params)

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    basenet_list = {'jrdb': Basenet_collective}
    gcnnet_list = {'Block': Block,
                   'distance': Distance, }

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.module_name]
        model = GCNnet(cfg)
        # # Load backbone
        # model.loadmodel(cfg.stage1_model_path)
    else:
        assert (False)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)

    model.train()
    model.apply(set_bn_eval)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)

    train_list = {'jrdb': train_jrdb}
    test_list = {'jrdb': test_jrdb}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]

    if cfg.test_before_train:
        test_info = test(validation_loader, model, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result_activity = {'epoch': 0, 'activities_f1': 0}
    best_result_action = {'epoch': 0, 'actions_f1': 0}
    best_result_social_activity = {'epoch': 0, 'social_activities_f1': 0}
    best_result_group_det = {'epoch': 0, 'group_detection_acc': 0}
    best_result_overall = {'epoch': 0, 'overall_f1': 0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        save_flag = 1

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # One epoch of forward and backward
        train_info = train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info, cfg)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info, cfg)

            if test_info['activities_f1'] > best_result_activity['activities_f1']:
                best_result_activity = test_info
                save_flag = 1
            print_log(cfg.log_path,
                      'Best group activity F1: %.2f%% at epoch #%d.' % (
                          best_result_activity['activities_f1'], best_result_activity['epoch']))

            if test_info['actions_f1'] > best_result_action['actions_f1']:
                best_result_action = test_info
            print_log(cfg.log_path,
                      'Best action F1: %.2f%% at epoch #%d.' % (
                          best_result_action['actions_f1'], best_result_action['epoch']))

            if test_info['social_activities_f1'] > best_result_social_activity['social_activities_f1']:
                best_result_social_activity = test_info
            print_log(cfg.log_path,
                      'Best social activity F1: %.2f%% at epoch #%d.' % (
                          best_result_social_activity['social_activities_f1'], best_result_social_activity['epoch']))

            if test_info['group_detection_acc'] > best_result_group_det['group_detection_acc']:
                best_result_group_det = test_info
            print_log(cfg.log_path,
                      'Best group det F1: %.2f%% at epoch #%d.' % (
                          best_result_group_det['group_detection_acc'], best_result_group_det['epoch']))

            if test_info['overall_f1'] > best_result_overall['overall_f1']:
                best_result_overall = test_info
            print_log(cfg.log_path,
                      'Best overall F1: %.2f%% at epoch #%d.' % (
                          best_result_overall['overall_f1'], best_result_overall['epoch']))

            # Save model
            if cfg.training_stage == 2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
                    cfg.training_stage, epoch, test_info['activities_f1'])
                torch.save(state, filepath)
                print('results saved to:', cfg.result_path)
            # elif cfg.training_stage == 1:
            # if save_flag == 1:
            #     for m in model.modules():
            #         if isinstance(m, GCNnet):
            #             filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
            #                 cfg.training_stage, epoch, test_info['actions_f1'])
            #             m.savemodel(filepath)
            # save_flag = 0


def train_jrdb(data_loader, model, device, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    epoch_timer = Timer()

    action_result = [[], []]
    activity_result = [[], []]
    social_activity_result = [[], []]
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # forward
        actions_scores, activities_scores, relation_graphs, social_acty_scores = model(
            (batch_data[0], batch_data[1], batch_data[4], batch_data[7]))

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes, cfg.num_actions))
        social_activity_in = batch_data[5].reshape((batch_size, num_frames, cfg.num_boxes, cfg.num_social_activities))
        social_group_id_in = batch_data[7].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames, cfg.num_activities))
        activities_in = activities_in[:, 0]
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)

        actions_in_nopad = []
        social_activity_in_nopad = []
        social_group_id_in_nopad = []
        for b in range(batch_size):
            N = bboxes_num[b][0]
            actions_in_nopad.append(actions_in[b][0][:N])
            social_group_id_in_nopad.append(social_group_id_in[b][0][:N])

            this_social_acty_label, _ = generate_social_acty_labels(social_activity_in[b][0][:N],
                                                                    social_group_id_in[b][0][:N],
                                                                    is_train=True)
            if this_social_acty_label != []:
                social_activity_in_nopad += this_social_acty_label
            else:
                index = len(social_activity_in_nopad)
                arr1 = social_acty_scores[0:index]
                arr2 = social_acty_scores[index + 1:]
                social_acty_scores = torch.cat((arr1, arr2), dim=0)

        actions_in = torch.cat(actions_in_nopad, dim=0)  # ALL_N,
        social_activity_in = torch.stack(social_activity_in_nopad)  # ALL_N,

        # -------------------------------------------------------------------------------------------------------------
        # relation graph loss
        social_group_GT_matrix = []
        relation_predict = []
        for b in range(batch_size):
            this_social_group_GT = np.array(create_P_matrix(social_group_id_in_nopad[b])).reshape((1, -1))
            social_group_GT_matrix.append(this_social_group_GT)
        social_group_GT = np.hstack(social_group_GT_matrix)
        social_group_GT = torch.tensor(social_group_GT).to(device=device)
        # ## set diag to 1
        for b in range(batch_size):
            N = relation_graphs[b].shape[-1]
            this_relation_graph = relation_graphs[b]
            relation_predict.append(this_relation_graph.reshape(1, -1))
        # ##
        relation_graphs_cat = torch.cat(relation_predict, dim=1).reshape(1, -1)
        relation_graph_BCE_loss = F.binary_cross_entropy(relation_graphs_cat.float(), social_group_GT.float(),
                                                         weight=None)
        group_detection_loss = relation_graph_BCE_loss
        # eig loss
        eig_loss = get_eig_loss(relation_predict, social_group_GT_matrix, device)
        # -------------------------------------------------------------------------------------------------------------
        # Predict actions
        actions_loss = F.binary_cross_entropy_with_logits(actions_scores.float(), actions_in.float(), weight=None)
        actions_labels = generate_labels(actions_scores, cfg.action_threshold)  # B*T*N,
        actions_in = np.array(actions_in.int().cpu())
        for ii in range(actions_labels.shape[0]):
            action_result[0].append(actions_labels[ii])
            action_result[1].append(actions_in[ii])

        # -------------------------------------------------------------------------------------------------------------
        # predict social activities
        social_activities_loss = F.binary_cross_entropy_with_logits(social_acty_scores.float(),
                                                                    social_activity_in.float(), )
        social_activities_labels = generate_labels(social_acty_scores, cfg.social_activity_threshold)  # ALL_N,
        social_activity_in = np.array(social_activity_in.int().cpu())
        for ii in range(social_activities_labels.shape[0]):
            social_activity_result[0].append(social_activities_labels[ii])
            social_activity_result[1].append(social_activity_in[ii])
        # -------------------------------------------------------------------------------------------------------------
        # Predict activities
        activities_loss = F.binary_cross_entropy_with_logits(activities_scores.float(), activities_in.float())
        activities_labels = generate_labels(activities_scores, cfg.activity_threshold)  # B*T,
        activities_in = np.array(activities_in.int().cpu())
        for ii in range(activities_labels.shape[0]):
            activity_result[0].append(activities_labels[ii])
            activity_result[1].append(activities_in[ii])
        # -------------------------------------------------------------------------------------------------------------
        # Total loss
        total_loss = cfg.activities_loss_weight * activities_loss + \
                     cfg.actions_loss_weight * actions_loss + \
                     cfg.relation_graph_loss_weight * group_detection_loss + \
                     cfg.social_activities_loss_weight * social_activities_loss + \
                     0.1 * eig_loss

        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # calculate P, R, F1
    action_p, action_r, action_f1, _ = precision_recall_fscore_support(action_result[1], action_result[0],
                                                                       sample_weight=None, average='samples')
    activity_p, activity_r, activity_f1, _ = precision_recall_fscore_support(activity_result[1], activity_result[0],
                                                                             sample_weight=None, average='samples')
    social_activity_p, social_activity_r, social_activity_f1, _ = precision_recall_fscore_support(
        social_activity_result[1], social_activity_result[0], sample_weight=None, average='samples')

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_f1': activity_f1 * 100,
        'actions_f1': action_f1 * 100,
        'social_activities_f1': social_activity_f1 * 100,
        'overall_f1': (activity_f1 + action_f1 + social_activity_f1) / 3 * 100
    }

    return train_info


def test_jrdb(data_loader, model, device, epoch, cfg):
    model.eval()

    loss_meter = AverageMeter()

    action_result = [[], []]
    activity_result = [[], []]
    group_detection_result = {
        'prediction': [],
        'GT': [],
    }
    social_activity_result = {
        'social_acty_gt': [],
        'social_acty_predict': [],
        'group_det_gt': [],
        'group_det_predict': [],
    }

    epoch_timer = Timer()
    with torch.no_grad():
        for batch_data in data_loader:

            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes, cfg.num_actions))
            social_activity_in = batch_data[5].reshape(
                (batch_size, num_frames, cfg.num_boxes, cfg.num_social_activities))
            social_group_id_in = batch_data[7].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size, num_frames, cfg.num_activities))
            activities_in = activities_in[:, 0]
            bboxes_num = batch_data[4].reshape(batch_size, num_frames)

            # forward
            actions_scores, activities_scores, relation_graphs, social_acty_scores = model(
                (batch_data[0], batch_data[1], batch_data[4], batch_data[7]))

            actions_in_nopad = []
            social_group_id_in_nopad = []
            N = bboxes_num[0][0]
            actions_in_nopad.append(actions_in[0][0][:N])
            social_group_id_in_nopad.append(social_group_id_in[0][0][:N])

            social_activity_in_gt, group_detection_gt = generate_social_acty_labels(social_activity_in[0][0][:N],
                                                                                    social_group_id_in[0][0][:N],
                                                                                    is_train=False)

            actions_in = torch.cat(actions_in_nopad, dim=0)
            social_activity_result['social_acty_gt'].append(social_activity_in_gt)
            social_activity_result['group_det_gt'].append(group_detection_gt)

            # -------------------------------------------------------------------------------------------------------------
            # predict the social group
            this_social_group_GT = np.array(social_group_id_in_nopad[0].cpu())
            # social group predict by spectral clustering
            this_N = relation_graphs[0].shape[0]
            this_cluster = SpectralClusterer(
                min_clusters=max(int(this_N * 0.6), 1),
                max_clusters=int(this_N), )
            this_social_group_predict = this_cluster.predict(np.array(relation_graphs[0].cpu()))
            group_detection_result['prediction'].append(create_P_matrix(this_social_group_predict))
            group_detection_result['GT'].append(create_P_matrix(this_social_group_GT))

            this_social_group_predict = generate_list(create_P_matrix(this_social_group_predict))
            this_social_group_predict.sort()
            social_activity_result['group_det_predict'].append(this_social_group_predict)

            # -------------------------------------------------------------------------------------------------------------
            # Predict actions
            actions_loss = F.binary_cross_entropy_with_logits(actions_scores.float(),
                                                              actions_in.float(),
                                                              weight=None)
            actions_labels = generate_labels(actions_scores, cfg.action_threshold)  # B*T*N,
            actions_in = np.array(actions_in.int().cpu())
            for ii in range(actions_labels.shape[0]):
                action_result[0].append(actions_labels[ii])
                action_result[1].append(actions_in[ii])

            # -------------------------------------------------------------------------------------------------------------
            # predict social activities
            social_activities_labels = generate_labels(social_acty_scores, cfg.social_activity_threshold)  # ALL_N,
            social_activity_result['social_acty_predict'].append(social_activities_labels)
            # -------------------------------------------------------------------------------------------------------------
            # Predict activities
            activities_loss = F.binary_cross_entropy_with_logits(activities_scores.float(), activities_in.float())
            activities_labels = generate_labels(activities_scores, cfg.activity_threshold)  # B*T,
            activities_in = np.array(activities_in.int().cpu())
            for ii in range(activities_labels.shape[0]):
                activity_result[0].append(activities_labels[ii])
                activity_result[1].append(activities_in[ii])
            # -------------------------------------------------------------------------------------------------------------
            # Total loss
            total_loss = cfg.activities_loss_weight * activities_loss + \
                         cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)

            # break

        # calculate P, R, F1
        action_p, action_r, action_f1, _ = precision_recall_fscore_support(action_result[1], action_result[0],
                                                                           sample_weight=None, average='samples')
        activity_p, activity_r, activity_f1, _ = precision_recall_fscore_support(activity_result[1], activity_result[0],
                                                                                 sample_weight=None, average='samples')
        group_p, group_r, group_f = evaluate_group_detection(group_detection_result['prediction'],
                                                             group_detection_result['GT'])
        social_activity_p, social_activity_r, social_activity_f1 = evaluate_social_activity(
            social_activity_result['social_acty_gt'],
            social_activity_result['social_acty_predict'],
            social_activity_result['group_det_gt'],
            social_activity_result['group_det_predict']
        )

        test_info = {
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'activities_f1': activity_f1 * 100,
            'actions_f1': action_f1 * 100,
            'social_activities_f1': social_activity_f1 * 100,

            'activities_p': activity_p * 100,
            'actions_p': action_p * 100,
            'social_activities_p': social_activity_p * 100,

            'activities_r': activity_r * 100,
            'actions_r': action_r * 100,
            'social_activities_r': social_activity_r * 100,

            # 'group_detection_acc': group_detection_meter.avg * 100,
            'group_detection_acc': group_f * 100,
            'overall_f1': (activity_f1 + action_f1 + social_activity_f1) / 3 * 100
        }

        # save group detection result
        np.save(
            '/home/yanhaomin/project/panoramic_action/code/joint_learning_for_jrdb/result/group_detection_results/acc_%.2f%%.npy' % (
                    group_f * 100),
            group_detection_result)

        return test_info

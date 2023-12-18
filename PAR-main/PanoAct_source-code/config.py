import time
import os
import json

def readJson(p:str):
    with open(p,'r') as f:
        config=json.load(f)
    return config

class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280  # input image size
        self.batch_size = 32  # train batch size
        self.test_batch_size = 8  # test batch size
        self.num_boxes = 12  # max number of bounding boxes in each frame
        self.num_SGAT = 16

        self.max_action_len = 6
        self.max_social_activity_len = 6
        self.max_global_activity_len = 3

        self.action_num_list = [13546, 11931, 8021, 8202, 4560, 4381, 1273, 1284, 1038, 475, 441, 435, 367, 274, 225,
                                161, 138, 86, 145, 122, 62, 64, 29, 12, 46, 13, 10]
        self.social_activity_num_list = [10896, 9426, 2903, 5549, 43, 64, 515, 482, 457, 325, 376, 159, 263, 1, 151, 105,
                                         5, 72, 4, 52, 23, 36, 2, 1, 46, 1, 10, 6638, 1, 853, 276, 939]
        self.global_activity_num_list = [484, 285, 575, 249, 1150, 211, 115]

        self.myJson = readJson('config.json')

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = True
        self.device_list = "0,1,2,3"  # id list of gpus used for training

        # Dataset
        # assert (dataset_name in ['jrdb'])
        self.dataset_name = dataset_name
        self.data_path = self.myJson["Datasets"]["JRDB"]['data_path']
        self.annotation_path = self.myJson["Datasets"]["JRDB"]['annotation_path']
        self.test_seqs = [2, 7, 11, 16, 17, 25, 26]
        self.train_seqs = [s for s in range(0, 26) if s not in self.test_seqs]

        # Backbone
        self.backbone = 'inv3'
        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 57, 467  # output feature map size of backbone
        self.emb_features = 1056  # output feature map channel of backbone
        self.stage1_model_path = self.myJson["Datasets"]["JRDB"]["stage1_model_path"]

        # Activity Action
        self.num_actions = 6  # number of action categories
        self.num_activities = 5  # number of activity categories
        self.num_social_activities = 6
        # self.action_weights = []
        # self.social_activity_weights = []

        self.actions_loss_weight = 1
        self.relation_graph_loss_weight = 1
        self.social_activities_loss_weight = 1
        self.activities_loss_weight = 1

        self.action_threshold = 0.5
        self.social_activity_threshold = 0.5
        self.activity_threshold = 0.5

        # Sample
        self.num_frames = 1
        self.num_before = 5
        self.num_after = 4

        # GCN
        self.num_features_boxes = 1024
        self.num_features_relation = 256
        self.num_SGAT = 64
        self.num_graph = 16  # number of graphs
        self.num_graphs = 64
        self.num_features_pose = 6
        self.num_features_trace = 6
        self.num_features_relation_app = 256
        self.num_features_relation_mt = 128
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1  # number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  # distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 666
        self.train_learning_rate = 2e-4  # initial learning rate
        self.lr_plan = {}  # change learning rate in these epochs
        self.train_dropout_prob = 0.3  # dropout probability
        self.weight_decay = 0  # l2 weight decay

        self.max_epoch = 150  # max training epoch
        self.test_interval_epoch = 1

        # Exp
        self.training_stage = 1  # specify stage1 or stage2
        self.stage1_model_path = ''  # path of the base model, need to be set in stage2
        self.test_before_train = False
        self.exp_note = 'Group-Activity-Recognition'
        self.exp_name = None

        self.module_name = ''
        self.my_note = ''
        self.experiment_index = 0

    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s]<%s>' % (self.exp_note, time_str)

        self.result_path = 'result/%s' % self.exp_name
        self.log_path = 'result/%s/log.txt' % self.exp_name

        if need_new_folder:
            os.mkdir(self.result_path)

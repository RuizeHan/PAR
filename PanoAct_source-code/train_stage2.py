import sys
from config import *

sys.path.append("scripts")
from train_net import *

cfg = Config('jrdb')

cfg.device_list = "0,1"
cfg.training_stage = 2
cfg.stage1_model_path = cfg.myJson["Datasets"]["JRDB"]["stage1_model_path"]  # PATH OF THE BASE MODEL
cfg.train_backbone = False

cfg.image_size = 480, 3760
cfg.out_size = 57, 467
cfg.num_boxes = 60
cfg.num_frames = 1
cfg.num_graph = cfg.num_graphs = cfg.num_SGAT=16
cfg.tau_sqrt = True
cfg.pos_threshold = 0.2
cfg.num_actions = 27
cfg.num_activities = 7
cfg.num_social_activities = 32

cfg.actions_loss_weight = 5
cfg.relation_graph_loss_weight = 4
cfg.social_activities_loss_weight = 3
cfg.activities_loss_weight = 0.1

cfg.action_threshold = 0.6
cfg.social_activity_threshold = 0.6
cfg.activity_threshold = 0.5

cfg.module_name = 'Block'

cfg.batch_size = 4
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 2e-5
cfg.lr_plan = {30: 1e-5}
cfg.train_dropout_prob = 0.2
cfg.weight_decay = 1e-2
cfg.max_epoch = 30

# #########################
cfg.my_note = 'all sigmoid'
cfg.exp_note = 'JRDB-act'
train_net(cfg)
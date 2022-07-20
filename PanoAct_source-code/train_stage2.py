import sys
from config import *

sys.path.append("scripts")
from train_net import *

cfg = Config('jrdb')

cfg.device_list = "0,1"
cfg.training_stage = 2
cfg.stage1_model_path = '/home/yanhaomin/project/panoramic_action/code/joint_learning_for_jrdb/result/epoch6_66.04%.pth'  # PATH OF THE BASE MODEL
cfg.train_backbone = True

cfg.image_size = 480, 3760
cfg.out_size = 57, 467
cfg.num_boxes = 60

cfg.num_actions = 27
cfg.num_activities = 7
cfg.num_social_activities = 32

cfg.num_frames = 2
cfg.num_graph = 4
cfg.tau_sqrt = True
cfg.num_features_boxes = 32
cfg.pos_threshold = 0.2

cfg.actions_loss_weight = 5
cfg.relation_graph_loss_weight = 4
cfg.social_activities_loss_weight = 3
cfg.activities_loss_weight = 0.1

# cfg.actions_loss_weight = 1
# cfg.relation_graph_loss_weight = 1
# cfg.social_activities_loss_weight = 1
# cfg.activities_loss_weight = 1

cfg.action_threshold = 0.6
cfg.social_activity_threshold = 0.6
cfg.activity_threshold = 0.5

cfg.module_name = 'block'

cfg.batch_size = 6
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 2e-5
cfg.lr_plan = {30: 1e-5}
cfg.train_dropout_prob = 0.2
cfg.weight_decay = 1e-2
cfg.lr_plan = {}
cfg.max_epoch = 300

# #########################
cfg.action_features = 32
cfg.num_graph = cfg.num_graphs=cfg.num_SGAT=16
cfg.num_features_relation = cfg.num_features_gcn = 256
cfg.num_features_nodes = cfg.action_features + 2

cfg.my_note = 'all sigmoid'

cfg.exp_note = 'JRDB-act'
train_net(cfg)

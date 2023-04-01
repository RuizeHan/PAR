import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np

CAD_path = '/home/yanhaomin/project/panoramic_action/code/joint_learning/data/collective'
social_CAD_path = '/home/yanhaomin/project/panoramic_action/code/joint_learning/data/social_CAD'
seqs = [s for s in range(1, 45)]

data_s = {}
data_c = {}
for sid in seqs:
    path_s = social_CAD_path + '/' + str(sid) + '_annotations.txt'
    path_c = CAD_path + '/seq%02d/annotations.txt' % sid

    data_s[sid] = {}
    data_c[sid] = {}

    with open(path_s, mode='r') as f:
        for l in f.readlines():
            values = l[:-1].split('	')
            this_frame_id = int(values[0])
            this_box = int(values[1])
            this_box_2 = int(values[2])
            this_box_3 = int(values[3])
            this_box_4 = int(values[4])
            this_action = int(values[5])
            if this_frame_id % 10 == 1:
                if this_frame_id not in data_s[sid].keys():
                    data_s[sid][this_frame_id] = {}
                if this_box not in data_s[sid][this_frame_id].keys():
                    data_s[sid][this_frame_id][this_box] = {'action': this_action,
                                                            'box': [this_box, this_box_2, this_box_3, this_box_4]}

    with open(path_c, mode='r') as f:
        for l in f.readlines():
            values = l[:-1].split('	')
            this_frame_id = int(values[0])
            this_box = int(values[1])
            this_box_2 = int(values[2])
            this_box_3 = int(values[3])
            this_box_4 = int(values[4])
            this_action = int(values[5])
            if this_frame_id % 10 == 1:
                if this_frame_id not in data_c[sid].keys():
                    data_c[sid][this_frame_id] = {}
                if this_box not in data_c[sid][this_frame_id].keys():
                    data_c[sid][this_frame_id][this_box] = {'action': this_action,
                                                            'box': [this_box, this_box_2, this_box_3, this_box_4]}

all = 0
different = 0
lack = 0
for seq_key in data_s.keys():
    for frame_key in data_s[seq_key]:
        for box_key in data_s[seq_key][frame_key]:
            all += 1
            if data_s[seq_key][frame_key][box_key]['action']  != data_c[seq_key][frame_key][box_key]['action'] :
                # print('!')
                different += 1
print(all)
print(different)

all = 0
different = 0
lack = 0
for seq_key in data_c.keys():
    for frame_key in data_c[seq_key]:
        for box_key in data_c[seq_key][frame_key]:
            all += 1
            if seq_key not in data_s.keys() or frame_key not in data_s[seq_key].keys() or box_key not in \
                    data_s[seq_key][frame_key]:
                # print('!')
                # print('seq key:' + str(seq_key) + ' frame key:' + str(frame_key) + ' bbox key:' + str(box_key))
                lack += 1
            elif data_c[seq_key][frame_key][box_key]['action'] != data_s[seq_key][frame_key][box_key]['action']:
                different += 1
                print('seq:' + str(seq_key) + ' frame:' + str(frame_key) + ' bbox:' + str(box_key))
                print(str(data_c[seq_key][frame_key][box_key]) + ' ' + str(data_s[seq_key][frame_key][box_key]))
                print()

print(all)
print(different)
print(lack)

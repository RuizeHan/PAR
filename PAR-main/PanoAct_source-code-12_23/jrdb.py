import torch
from torch.utils import data
import torchvision.transforms as transforms
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
from config import readJson
from os.path import join

# <sequence_id>,<keyframe-id >,<[x1,y1,x2,y2]>,<social group id>,<action/activity id>,<no use>,<person id>
NUM_FRAMES = {0: 1728, 1: 580, 2: 1441, 3: 1009, 4: 1298, 5: 1448, 6: 864, 7: 467, 8: 1011, 9: 726, 10: 1730, 11: 874,
              12: 1156, 13: 725, 14: 1524, 15: 1091, 16: 460, 17: 1014, 18: 718,
              19: 853, 20: 1373, 21: 1737, 22: 868, 23: 873, 24: 432, 25: 505, 26: 1442}


def read_excel(excel_dict_path):
    excel_path_list = sorted(os.listdir(excel_dict_path))
    data = {}

    for path in excel_path_list:
        excel_path = excel_dict_path + path
        # read excel file
        excel = pd.read_excel(excel_path)  # reading file
        seq_id_list = excel["seq_id"].tolist()
        frame_id_list = excel["frame_id"].tolist()
        global_labels_list = excel["global_labels"].to_list()

        # conversion new labels from str to int
        for i in range(len(global_labels_list)):
            new_label_int = str(global_labels_list[i]).split(' ')
            temp = []
            for k in new_label_int:
                if k != '' and k != ' ' and k != 'nan':
                    temp.append(int(k) - 1)

            global_labels_list[i] = temp

        # build data dictionary
        for i in range(len(seq_id_list)):
            if seq_id_list[i] in [int(i) for i in range(0, 27)]:
                seq_id_list[i] = int(seq_id_list[i])
                if seq_id_list[i] not in data:
                    data[seq_id_list[i]] = {}
                if frame_id_list[i] not in data[seq_id_list[i]]:
                    data[seq_id_list[i]][frame_id_list[i]] = {'global_labels': global_labels_list[i],
                                                              'single_global_label': global_labels_list[i][0]}

    return data


def one_hot(labels, num_categories):
    result = [0 for _ in range(num_categories)]
    for label in labels:
        result[label] = 1
    return result


def jrdb_read_dataset_new(path, seqs, num_actions, num_activities, num_social_activities):
    myjson = readJson('config.json')
    action_path = path + '/gt_action.txt'
    activity_path = path + '/gt_activity.txt'
    group_path = path + '/gt_group.txt'
    excel_path = myjson["Datasets"]["JRDB"]['excel_path']

    # read action annotation and build dict
    action_dict = {}
    with open(action_path, mode='r') as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            this_seq_id = int(values[0])
            this_frame_id = int(values[1])
            this_person_id = int(values[9])
            this_action = int(values[7]) - 1

            if this_seq_id in seqs:
                if this_seq_id not in action_dict:
                    action_dict[this_seq_id] = {}
                if this_frame_id not in action_dict[this_seq_id]:
                    action_dict[this_seq_id][this_frame_id] = {}
                if this_person_id not in action_dict[this_seq_id][this_frame_id]:
                    action_dict[this_seq_id][this_frame_id][this_person_id] = {
                        'action': [],
                        'single_action': this_action,
                    }
                action_dict[this_seq_id][this_frame_id][this_person_id]['action'].append(this_action)

    # read group detection annotation and build dict
    group_dict = {}
    with open(group_path, mode='r') as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            this_seq_id = int(values[0])
            this_frame_id = int(values[1])
            this_person_id = int(values[9])
            this_group_id = int(values[6])

            if this_seq_id in seqs:
                if this_seq_id not in group_dict:
                    group_dict[this_seq_id] = {}
                if this_frame_id not in group_dict[this_seq_id]:
                    group_dict[this_seq_id][this_frame_id] = {}
                if this_person_id not in group_dict[this_seq_id][this_frame_id]:
                    group_dict[this_seq_id][this_frame_id][this_person_id] = {
                        'group_id': this_group_id,
                    }

    # read excel file and get the global activity
    global_activity_dict = read_excel(excel_path)

    # read social activity annotation and build dict
    social_activity_dict = {}
    with open(activity_path, mode='r') as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            this_seq_id = int(values[0])
            this_frame_id = int(values[1])
            this_person_id = int(values[9])
            this_social_activity = int(values[7]) - 1

            x1, y1, x2, y2 = (int(values[i]) for i in range(2, 6))
            H, W = (480, 3760)
            this_bbox = (y1 / H, x1 / W, y2 / H, x2 / W)

            if this_seq_id in seqs:
                if this_seq_id not in social_activity_dict:
                    social_activity_dict[this_seq_id] = {}
                if this_frame_id not in social_activity_dict[this_seq_id]:
                    social_activity_dict[this_seq_id][this_frame_id] = {}
                if this_person_id not in social_activity_dict[this_seq_id][this_frame_id]:
                    social_activity_dict[this_seq_id][this_frame_id][this_person_id] = {
                        'social_activity': [],
                        'single_social_activity': this_social_activity,
                        'bbox': this_bbox,
                    }
                social_activity_dict[this_seq_id][this_frame_id][this_person_id]['social_activity'].append(
                    this_social_activity)

    # build the final data dict with everything according to the above dicts
    data_all = {}
    for seq_id in social_activity_dict.keys():
        for frame_id in social_activity_dict[seq_id].keys():
            for person_id in social_activity_dict[seq_id][frame_id].keys():

                if person_id in action_dict[seq_id][frame_id].keys():
                    if seq_id not in data_all:
                        data_all[seq_id] = {}
                    if frame_id not in data_all[seq_id]:
                        data_all[seq_id][frame_id] = {
                            'frame_id': frame_id,
                            'group_activity': global_activity_dict[seq_id][15]['single_global_label'],
                            'social_activity': [],
                            'actions': [],
                            'person_id': [],
                            'social_group_id': [],
                            'bboxes': [],
                            'group_activity_all': one_hot(global_activity_dict[seq_id][15]['global_labels'],
                                                        num_activities),
                            'social_activity_all': [],
                            'actions_all': [],
                        }

                    data_all[seq_id][frame_id]['social_activity'].append(
                        social_activity_dict[seq_id][frame_id][person_id]['single_social_activity'])
                    data_all[seq_id][frame_id]['actions'].append(
                        action_dict[seq_id][frame_id][person_id]['single_action'])

                    data_all[seq_id][frame_id]['social_activity_all'].append(
                        one_hot(social_activity_dict[seq_id][frame_id][person_id]['social_activity'],
                                num_social_activities))
                    data_all[seq_id][frame_id]['actions_all'].append(
                        one_hot(action_dict[seq_id][frame_id][person_id]['action'], num_actions))

                    data_all[seq_id][frame_id]['person_id'].append(person_id)
                    data_all[seq_id][frame_id]['social_group_id'].append(
                        group_dict[seq_id][frame_id][person_id]['group_id'])
                    data_all[seq_id][frame_id]['bboxes'].append(
                        social_activity_dict[seq_id][frame_id][person_id]['bbox'])
                else:
                    pass
    

    return data_all


def jrdb_all_frames(anns):
    return [(s, f) for s in anns for f in anns[s]]


class JRDB_Dataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """

    def __init__(self, num_actions, num_activities, num_social_activities, anns, frames, images_path, image_size,
                 feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_finetune=False):
        self.anns = anns
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.is_training = is_training
        self.is_finetune = is_finetune

        self.num_actions = num_actions
        self.num_activities = num_activities
        self.num_social_activities = num_social_activities

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """

        select_frames = self.get_frames(self.frames[index])

        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):

        sid, src_fid = frame

        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid, src_fid + self.num_frames - 1)
                return [(sid, src_fid, fid)]

            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid, src_fid + self.num_frames)]

        else:
            if self.is_training:
                sample_frames = random.sample(range(src_fid - self.num_frames, src_fid + self.num_frames + 1), 1)
                return [(sid, src_fid, fid) for fid in sample_frames]
            else:
                return [(sid, src_fid, src_fid)]

    def one_hot(self, labels, num_categories):
        result = [0 for _ in range(num_categories)]
        for label in labels:
            result[label] = 1
        return result

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        social_activity = []
        person_id = []
        social_group_id = []
        bboxes_num = []
        seq_id = []
        frame_id = []
        table = []

        # one_hot = torch.nn.functional.one_hot(unique_actions, cfg.num_actions).float()
        zero_action = [0 for _ in range(self.num_actions)]
        zero_social_activity = [0 for _ in range(self.num_social_activities)]

        seq_names = sorted(os.listdir(self.images_path))
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            this_image_path = self.images_path + '/' + seq_names[int(sid)] + '/' + str(fid).zfill(6) + ".jpg"
            if os.path.exists(this_image_path):
                img = Image.open(this_image_path)
            else:
                img = Image.open(self.images_path + '/' + seq_names[int(sid)] + '/' + str(src_fid).zfill(6) + ".jpg")
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            temp_boxes = []
            temp_table = []
            for box in self.anns[sid][src_fid]['bboxes']:
                y1, x1, y2, x2 = box
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes.append((w1, h1, w2, h2))
                temp_table.append({'img_path': this_image_path, 'boxes_coord': (w1, h1, w2, h2)})

            temp_actions = self.anns[sid][src_fid]['actions_all'][:]
            temp_social_activity = self.anns[sid][src_fid]['social_activity_all'][:]
            temp_person_id = self.anns[sid][src_fid]['person_id'][:]
            temp_social_group_id = self.anns[sid][src_fid]['social_group_id'][:]
            bboxes_num.append(len(temp_boxes))
            temp_sid = []
            temp_fid = []
            for kk in range(len(temp_person_id)):
                temp_sid.append(sid)
                temp_fid.append(fid)

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(zero_action)
                temp_social_activity.append(zero_social_activity)
                temp_person_id.append(-1)
                temp_social_group_id.append(-1)
                temp_fid.append(-1)
                temp_sid.append(-1)
                temp_table.append({})

            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            social_activity.append(temp_social_activity)
            person_id.append(temp_person_id)
            social_group_id.append(temp_social_group_id)
            seq_id.append(temp_sid)
            frame_id.append(temp_fid)
            table.append(temp_table)
            activities.append(self.anns[sid][src_fid]['group_activity_all'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32).reshape((-1, self.num_activities))
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float).reshape((-1, self.num_boxes, 4))
        actions = np.array(actions, dtype=np.int32).reshape((-1, self.num_boxes, self.num_actions))
        social_activity = np.array(social_activity, dtype=np.int32).reshape(
            (-1, self.num_boxes, self.num_social_activities))
        person_id = np.array(person_id, dtype=np.int32).reshape(-1, self.num_boxes)
        seq_id = np.array(seq_id, dtype=np.int32).reshape(-1, self.num_boxes)
        frame_id = np.array(frame_id, dtype=np.int32).reshape(-1, self.num_boxes)
        social_group_id = np.array(social_group_id, dtype=np.int32).reshape(-1, self.num_boxes)

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        social_activity = torch.from_numpy(social_activity).long()
        person_id = torch.from_numpy(person_id).long()
        seq_id = torch.from_numpy(seq_id).long()
        frame_id = torch.from_numpy(frame_id).long()
        social_group_id = torch.from_numpy(social_group_id).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()

        return images, bboxes, actions, activities, bboxes_num, social_activity, \
               person_id, social_group_id, seq_id, frame_id

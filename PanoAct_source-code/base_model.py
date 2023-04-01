import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from roi_align.roi_align import CropAndResize  # crop_and_resize module
from self_attention_cv import AxialAttentionBlock


class Basenet_collective(nn.Module):
    """
    main module of base model for collective dataset
    """

    def __init__(self, cfg):
        super(Basenet_collective, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)
        self.attention = AxialAttentionBlock(in_channels=D, dim=K, heads=8)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict': self.fc_emb_1.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        EPS = 1e-5

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4

        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,

        # self attention
        boxes_features_all = boxes_features_all.reshape(B * T * MAX_N, D, K, K)  # B*T*MAX_N, D,K,K
        boxes_features_all = self.attention(boxes_features_all)
        boxes_features_all = boxes_features_all.reshape(B * T, MAX_N, -1)  # B*T,MAX_N, D*K*K


        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all = F.relu(boxes_features_all)
        boxes_features_all = self.dropout_emb_1(boxes_features_all)

        actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B * T, )  # B*T,

        for bt in range(B * T):
            N = bboxes_num_in[bt]
            boxes_features = boxes_features_all[bt, :N, :].reshape(1, N, NFB)  # 1,N,NFB

            boxes_states = boxes_features

            NFS = NFB

            # Predict actions
            boxes_states_flat = boxes_states.reshape(-1, NFS)  # 1*N, NFS
            actn_score = self.fc_actions(boxes_states_flat)  # 1*N, actn_num
            actions_scores.append(actn_score)

            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)  # 1, NFS
            boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)  # 1, NFS
            acty_score = self.fc_activities(boxes_states_pooled_flat)  # 1, acty_num
            activities_scores.append(acty_score)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B*T,acty_num

        return actions_scores, activities_scores

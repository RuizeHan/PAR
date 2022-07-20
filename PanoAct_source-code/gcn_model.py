import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from roi_align.roi_align import CropAndResize  # crop_and_resize module


class GCN_w_position_embedding(nn.Module):
    def __init__(self, cfg):
        super(GCN_w_position_embedding, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.NFP = 256

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG + self.NFP, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG + self.NFP, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        device = graph_boxes_features.device

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # My position
        boxes_distance_foot = calc_foot_distance(boxes_in_flat).to(device)
        position_mask = (boxes_distance_foot > (pos_threshold * OW))
        boxes_distance_foot_normalized = normalize_distance(boxes_in_flat, boxes_distance_foot)
        boxes_distance_foot_normalized = 1 / boxes_distance_foot_normalized
        boxes_distance_foot_normalized = torch.nan_to_num(boxes_distance_foot_normalized, 0, 1)
        boxes_distance_foot_normalized = torch.sigmoid(boxes_distance_foot_normalized)

        # calculate position embedding
        boxes_in_flat = boxes_in_flat.reshape(-1, 4)
        xy_foot = torch.zeros((N, 2)).to(device)
        xy_foot[:, 0] = (boxes_in_flat[:, 0] + boxes_in_flat[:, 2]) / 2
        xy_foot[:, 1] = boxes_in_flat[:, 3]
        xy_foot = xy_foot.reshape(N, 2)
        boxes_embedding = sincos_encoding_2d(xy_foot, self.NFP).reshape(1, N, self.NFP)

        relation_graph_list = []
        graph_boxes_features_list = []
        for i in range(NG):
            features_plus_position = torch.cat((boxes_embedding, graph_boxes_features), dim=-1)
            # features_plus_position=graph_boxes_features
            graph_boxes_features_theta = self.fc_rn_theta_list[i](features_plus_position)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](features_plus_position)  # B,N,NFR

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N
            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR + self.NFP)
            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph
            relation_graph = relation_graph.reshape(B, N, N)
            relation_graph[position_mask] = -float('inf')

            relation_graph_w_softmax = torch.softmax(relation_graph, dim=2)
            relation_graph_list.append(relation_graph_w_softmax)

            relation_graph_for_convolution = relation_graph_w_softmax
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph_for_convolution, graph_boxes_features))  # B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list), dim=0)  # B, N, NFG
        mean_relation_graph = torch.mean(torch.stack(relation_graph_list), dim=0)
        # --------------------------------------------------------------------------------------------------------------
        # final_graph = 0.7 * mean_relation_graph + 0.5 * boxes_distance_foot_normalized
        final_graph = mean_relation_graph


        for i in range(N):
            for j in range(N):
                if final_graph[0][i][j] > 1:
                    final_graph[0][i][j] = 1
        # --------------------------------------------------------------------------------------------------------------
        return graph_boxes_features, final_graph


class G2O(nn.Module):
    def __init__(self, cfg):
        super(G2O, self).__init__()
        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.NFP = 256

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, graph_boxes_features):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        device = graph_boxes_features.device

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        relation_graph_list = []
        graph_boxes_features_list = []
        for i in range(NG):
            features_plus_position = graph_boxes_features
            graph_boxes_features_theta = self.fc_rn_theta_list[i](features_plus_position)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](features_plus_position)  # B,N,NFR

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N
            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR + self.NFP)
            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph
            relation_graph = relation_graph.reshape(B, N, N).sum(2)

            relation_graph_w_softmax = torch.softmax(relation_graph, dim=1)
            # ln = torch.nn.LayerNorm([N, N], elementwise_affine=False)
            # relation_graph_w_ln = ln(relation_graph_w_softmax.reshape(N, N)).reshape(B, N, N)
            relation_graph_list.append(relation_graph_w_softmax)

            relation_graph_for_convolution = relation_graph_w_softmax
            one_graph_boxes_features = self.fc_gcn_list[i](
                (relation_graph_for_convolution[..., None] * graph_boxes_features).sum(dim=1,
                                                                                       keepdim=True))  # B, 1, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_features = torch.mean(torch.stack(graph_boxes_features_list), dim=0)  # B, 1, NFG
        # graph_features, _ = torch.max(torch.stack(graph_boxes_features_list), dim=0)  # B, 1, NFG
        # out_put_features = graph_features + torch.mean(graph_boxes_features.reshape(N, -1), dim=0)
        return graph_features

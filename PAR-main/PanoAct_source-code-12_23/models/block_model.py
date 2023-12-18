import torch
from gcn_model import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from self_attention_cv import AxialAttentionBlock
from GAT.models import GAT
from spectral_cluster.spectralcluster import SpectralClusterer
from torchvision.ops import roi_align
from utils import *
from my_utils.distance import Distance


class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
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
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.global_fc_emb_1 = nn.Linear(D, NFB)

        # self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        # self.gcn_list_1 = torch.nn.ModuleList([GCN_w_position_embedding(self.cfg) for i in range(self.cfg.gcn_layers)])
        self.gcn_ind = GCN_w_position_embedding(self.cfg)

        # self.gcn_ind2group = GCN_w_position_embedding(self.cfg)

        self.gcn_group = G2O(self.cfg)

        self.gcn_group2global = G2O(self.cfg)
        self.gcn_ind2global = G2O(self.cfg)

        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.FFN_actions = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(256, self.cfg.num_actions)
        )
        self.FFN_social_activities = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(256, self.cfg.num_social_activities)
        )
        self.FFN_activities = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(512, self.cfg.num_activities)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
        }
        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        print('Load model states from: ', filepath)

    def predict_social_activity_score_train(self, social_group_id, boxes_states):
        # calculate the pooled feature for each bbox according to the gt group they belong to
        social_group_id = social_group_id.reshape(-1)
        social_group_gt = create_P_matrix(np.array(social_group_id.cpu()))
        social_group_gt = generate_list(social_group_gt)
        social_group_gt.sort()
        group_states_all = []
        for this_group in social_group_gt:
            group_states_list_this = []
            for person_id in this_group:
                group_states_list_this.append(boxes_states[person_id])
            group_states_this = torch.stack(group_states_list_this)
            # group_states_this = group_states_this.mean(dim=0)
            group_states_this = self.gcn_group(group_states_this[None, ...])
            group_states_all.append(group_states_this)

        # in case there is no group in a image:
        if not group_states_all:
            group_states_all.append(boxes_states[0])
        group_states_all = torch.stack(group_states_all)

        return group_states_all

    def predict_social_activity_score_test(self, relation_graph, boxes_states):
        this_N = relation_graph.shape[-1]
        this_cluster = SpectralClusterer(
            min_clusters=max(int(this_N * 0.6), 1),
            max_clusters=int(this_N),
            autotune=None,
            laplacian_type=None,
            refinement_options=None)
        # use clustering to get the groups
        social_group_predict = this_cluster.predict(np.array(relation_graph.cpu()))
        social_group_predict = create_P_matrix(np.array(social_group_predict))
        social_group_predict = generate_list(social_group_predict)
        social_group_predict.sort()

        group_states_all = []
        for this_group in social_group_predict:
            group_states_list_this = []
            for person_id in this_group:
                group_states_list_this.append(boxes_states[person_id])
            group_states_this = torch.stack(group_states_list_this)
            group_states_this = self.gcn_group(group_states_this[None, ...])
            # group_states_this = group_states_this.mean(dim=0)
            group_states_all.append(group_states_this)

        # in case there is no group in a image:
        if not group_states_all:
            group_states_all.append(boxes_states[0])
        group_states_all = torch.stack(group_states_all)
        return group_states_all

    # def predict_social_activity_score_test_w_gt(self, boxes_states, social_group_id):
    #     social_group_id = social_group_id.reshape(-1)
    #     social_group_gt = create_P_matrix(np.array(social_group_id.cpu()))
    #     social_group_gt = generate_list(social_group_gt)
    #     social_group_gt.sort()
    #
    #     group_states_all = []
    #     for this_group in social_group_gt:
    #         group_states_list_this = []
    #         for person_id in this_group:
    #             group_states_list_this.append(boxes_states[person_id])
    #         group_states_this = torch.stack(group_states_list_this)
    #         group_states_this = self.gcn_group(group_states_this[None, ...])
    #         # group_states_this = group_states_this.mean(dim=0)
    #         group_states_all.append(group_states_this)
    #
    #     # in case there is no group in a image:
    #     if not group_states_all:
    #         group_states_all.append(boxes_states[0])
    #     group_states_all = torch.stack(group_states_all)
    #     return group_states_all

    def forward(self, batch_data):
        global graph_boxes_features
        images_in, boxes_in, bboxes_num_in, bboxes_social_group_id_in, seq_id, frame_id = batch_data

        # read config parameters
        B = images_in.shape[0]  # image_in: B * T * 3 * H *W
        T = images_in.shape[1]  # equals to 1 here
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

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

        # # self attention
        # boxes_features_all = boxes_features_all.reshape(B * T * MAX_N, D, K, K)  # B*T*MAX_N, D,K,K
        # boxes_features_all = self.attention(boxes_features_all)
        boxes_features_all = boxes_features_all.reshape(B * T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        boxes_in = boxes_in.reshape(B, T, MAX_N, 4)
        bboxes_social_group_id_in = bboxes_social_group_id_in.reshape(B, T, MAX_N, 1)

        actions_scores = []
        social_acty_scores = []
        activities_scores = []
        relation_graphs = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T
        seq_ids = []
        frame_ids = []
        boxes_coords = []

        for b in range(B):
            N = bboxes_num_in[b][0]

            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T * N, NFB)  # 1,T,N,NFB
            seq_no_pad = seq_id[b, :, :N].reshape(-1, 1)
            frame_no_pad = frame_id[b, :, :N].reshape(-1, 1)

            boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            bboxes_social_group_id = bboxes_social_group_id_in[b, :, :N, :].reshape(1, T * N)  # T*N, 4

            # Node graph, for individual
            graph_boxes_features, relation_graph = self.gcn_ind(boxes_features, boxes_positions)

            # graph attention module
            graph_boxes_features = graph_boxes_features.reshape(1, T * N, NFB)
            # graph_boxes_features, adjacency_matrix = self.graph_attention(graph_boxes_features, relation_graph)

            # handle the relation graph
            relation_graph = relation_graph.reshape(relation_graph.shape[1], relation_graph.shape[1])
            relation_graphs.append(relation_graph)
            # social_group_id = bboxes_social_group_id.reshape(-1)
            # relation_graph_gt = create_P_matrix(np.array(social_group_id.cpu()))
            # relation_graph_gt = torch.from_numpy(relation_graph_gt).to(graph_boxes_features.device)
            # relation_graphs.append(relation_graph_gt)

            # cat graph_boxes_features with boxes_features
            boxes_features = boxes_features.reshape(1, T * N, NFB)
            boxes_states = graph_boxes_features + boxes_features  # 1, T*N, NFG
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            boxes_states = boxes_states.reshape(T * N, NFS)

            # Predict social activities
            if self.training:
                boxes_states_for_social_activity = self.predict_social_activity_score_train(
                    bboxes_social_group_id, boxes_states.reshape(N * T, NFS))
            else:
                boxes_states_for_social_activity = self.predict_social_activity_score_test(
                    relation_graph, boxes_states.reshape(N * T, NFS))
            boxes_states_for_social_activity = boxes_states_for_social_activity.reshape(-1, NFS)

            # Predict activities
            # 1.not use global feature:
            # boxes_states_pooled = torch.mean(boxes_states, dim=0)
            boxes_states_pooled = self.gcn_ind2global(boxes_states.reshape(1, T * N, NFS))
            boxes_states_pooled_1 = boxes_states_pooled.reshape(1, NFS)

            # group_states_pooled = torch.mean(boxes_states_for_social_activity, dim=0).reshape(1, NFS)
            group_states_pooled = self.gcn_group2global(boxes_states_for_social_activity.reshape(1, -1, NFS)).reshape(1,
                                                                                                                      NFS)

            global_f = boxes_states_pooled_1 + group_states_pooled
            acty_score = self.FFN_activities(global_f)

            # Predict actions
            actn_score = self.FFN_actions(boxes_states.reshape(-1, NFS) + global_f)  # T,N, actn_num
            social_acty_score = self.FFN_social_activities(boxes_states_for_social_activity + global_f)

            actions_scores.append(actn_score)
            social_acty_scores.append(social_acty_score)
            activities_scores.append(acty_score)
            
            seq_ids.append(seq_no_pad)
            frame_ids.append(frame_no_pad)
            boxes_coords.append(boxes_positions)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        social_acty_scores = torch.cat(social_acty_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        
        seq_ids = torch.cat(seq_ids, dim=0)
        frame_ids = torch.cat(frame_ids, dim=0)
        boxes_coords = torch.cat(boxes_coords, dim=0)

        return actions_scores, activities_scores, relation_graphs, social_acty_scores, seq_ids, frame_ids, boxes_coords
